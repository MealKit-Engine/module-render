#include "mealkit/render/Renderer.h"

#include "mealkit/core/Log.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wincodec.h>
#include <wrl/client.h>

#include "g_ForwardPS.h"
#include "g_ForwardVS.h"

#if defined(TRACY_ENABLE)
#include <optional>
#include <tracy/TracyD3D12.hpp>
#endif

using Microsoft::WRL::ComPtr;

namespace mealkit::render {
namespace {

constexpr u32 kFramesInFlight = 2;
constexpr u32 kBackBufferCount = 3;
constexpr u64 kUploadBytesPerFrame = 8ull << 20; // 인스턴스 약 17만 개분
constexpr DXGI_FORMAT kSwapchainFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
constexpr DXGI_FORMAT kRtvFormat = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB; // 셰이더는 선형으로 쓰고 하드웨어가 감마 인코딩
constexpr DXGI_FORMAT kOverlayFormat = DXGI_FORMAT_R8G8B8A8_UNORM;  // UI는 이미 sRGB 색이므로 변환하지 않는다
constexpr DXGI_FORMAT kDepthFormat = DXGI_FORMAT_D32_FLOAT;

struct alignas(16) FrameConstants {
    f32 viewProj[16];
    f32 cameraPos[4];
    f32 sunDir[4];
    f32 sunColor[4];
    f32 skyColor[4];
    f32 groundColor[4];
    f32 fogColor[4];
};
static_assert(sizeof(FrameConstants) == 160);

ID3D12Device* g_errorDevice = nullptr;

void checkHr(HRESULT hr, const char* expr, const char* file, int line) {
    if (SUCCEEDED(hr)) return;
    HRESULT reason = S_OK;
    if (g_errorDevice && (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET))
        reason = g_errorDevice->GetDeviceRemovedReason();
    MEALKIT_FATAL("D3D12 call failed (hr=0x%08X, device removed reason=0x%08X)\n%s\n%s:%d", static_cast<unsigned>(hr),
                static_cast<unsigned>(reason), expr, file, line);
}
#define HR(expr) checkHr((expr), #expr, __FILE__, __LINE__)

std::string narrow(const wchar_t* w) {
    const int len = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    std::string out(len > 0 ? len - 1 : 0, '\0');
    if (len > 1) WideCharToMultiByte(CP_UTF8, 0, w, -1, out.data(), len, nullptr, nullptr);
    return out;
}

std::wstring widen(const char* s) {
    const int len = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
    std::wstring out(len > 0 ? len - 1 : 0, L'\0');
    if (len > 1) MultiByteToWideChar(CP_UTF8, 0, s, -1, out.data(), len);
    return out;
}

D3D12_HEAP_PROPERTIES heapProperties(D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES p{};
    p.Type = type;
    p.CreationNodeMask = 1;
    p.VisibleNodeMask = 1;
    return p;
}

D3D12_RESOURCE_DESC bufferDesc(u64 size) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = size;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = DXGI_FORMAT_UNKNOWN;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return d;
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                                  D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = resource;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    return b;
}

void __stdcall onDebugMessage(D3D12_MESSAGE_CATEGORY, D3D12_MESSAGE_SEVERITY severity, D3D12_MESSAGE_ID,
                              LPCSTR description, void*) {
    if (severity <= D3D12_MESSAGE_SEVERITY_ERROR) MEALKIT_ERROR("[D3D12] %s", description);
    else if (severity == D3D12_MESSAGE_SEVERITY_WARNING) MEALKIT_WARN("[D3D12] %s", description);
}

struct GpuMesh {
    ComPtr<ID3D12Resource> vertexBuffer;
    ComPtr<ID3D12Resource> indexBuffer;
    D3D12_VERTEX_BUFFER_VIEW vbv{};
    D3D12_INDEX_BUFFER_VIEW ibv{};
    u32 indexCount = 0;
};

} // namespace

struct Renderer::Impl {
    HWND hwnd = nullptr;
    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;

    ComPtr<IDXGISwapChain3> swapchain;
    HANDLE frameLatencyWaitable = nullptr;
    UINT swapchainFlags = 0;
    bool tearingSupported = false;

    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    ComPtr<ID3D12DescriptorHeap> dsvHeap;
    UINT rtvStride = 0;
    ComPtr<ID3D12Resource> backBuffers[kBackBufferCount];
    ComPtr<ID3D12Resource> depthBuffer;

    struct Frame {
        ComPtr<ID3D12CommandAllocator> allocator;
        u64 fenceValue = 0;
    } frames[kFramesInFlight];
    ComPtr<ID3D12GraphicsCommandList> cmd;

    ComPtr<ID3D12Fence> fence;
    HANDLE fenceEvent = nullptr;
    u64 fenceCounter = 0;

    ComPtr<ID3D12Resource> uploadBuffer;
    u8* uploadCpu = nullptr;
    D3D12_GPU_VIRTUAL_ADDRESS uploadGpu = 0;
    u64 uploadOffset = 0;
    u64 uploadLimit = 0;
    bool uploadOverflowWarned = false;

    ComPtr<ID3D12RootSignature> rootSignature;
    ComPtr<ID3D12PipelineState> pipeline;
    std::vector<GpuMesh> meshes;

    u32 width = 0, height = 0;
    u32 frameIndex = 0;
    UINT backBufferIndex = 0;
    bool vsync = true;
    bool inFrame = false;
    std::string adapterName;
    bool uma = false;
    RenderStats stats;
    std::string capturePath;
#if defined(TRACY_ENABLE)
    tracy::D3D12QueueCtx* tracyGpu = nullptr;          // GPU 시간 측정 (프로파일 빌드)
    std::optional<tracy::D3D12ZoneScope> tracyFrameZone;
#endif

    u64 signal() {
        ++fenceCounter;
        HR(queue->Signal(fence.Get(), fenceCounter));
        return fenceCounter;
    }

    void waitForFence(u64 value) {
        if (fence->GetCompletedValue() < value) {
            HR(fence->SetEventOnCompletion(value, fenceEvent));
            WaitForSingleObject(fenceEvent, INFINITE);
        }
    }

    void waitIdle() { waitForFence(signal()); }

    // 이번 프레임 구간에서 256바이트 정렬로 잘라 준다. 공간이 없으면 0.
    D3D12_GPU_VIRTUAL_ADDRESS allocUpload(u64 size, void** cpu) {
        const u64 aligned = (uploadOffset + 255) & ~255ull;
        if (aligned + size > uploadLimit) return 0;
        uploadOffset = aligned + size;
        *cpu = uploadCpu + aligned;
        return uploadGpu + aligned;
    }

    bool pickAdapter() {
        for (UINT i = 0;; ++i) {
            ComPtr<IDXGIAdapter1> candidate;
            if (factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                    IID_PPV_ARGS(&candidate)) == DXGI_ERROR_NOT_FOUND)
                break;
            DXGI_ADAPTER_DESC1 desc;
            candidate->GetDesc1(&desc);
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            if (SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr))) {
                adapter = candidate;
                adapterName = narrow(desc.Description);
                return true;
            }
        }
        return false;
    }

    // 정적 버퍼: DEFAULT 힙에 만들고 임시 업로드 버퍼에서 복사한다 (로딩 시점 전용).
    ComPtr<ID3D12Resource> createStaticBuffer(const void* data, u64 size) {
        const D3D12_HEAP_PROPERTIES defaultHeap = heapProperties(D3D12_HEAP_TYPE_DEFAULT);
        const D3D12_HEAP_PROPERTIES uploadHeap = heapProperties(D3D12_HEAP_TYPE_UPLOAD);
        const D3D12_RESOURCE_DESC desc = bufferDesc(size);

        ComPtr<ID3D12Resource> buffer;
        HR(device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                           nullptr, IID_PPV_ARGS(&buffer)));
        ComPtr<ID3D12Resource> staging;
        HR(device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &desc,
                                           D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&staging)));
        void* mapped = nullptr;
        const D3D12_RANGE noRead{0, 0};
        HR(staging->Map(0, &noRead, &mapped));
        std::memcpy(mapped, data, size);
        staging->Unmap(0, nullptr);

        ComPtr<ID3D12CommandAllocator> allocator;
        HR(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));
        ComPtr<ID3D12GraphicsCommandList> list;
        HR(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                     IID_PPV_ARGS(&list)));
        // 버퍼는 COMMON에서 COPY_DEST로, 실행 후 다시 읽기 상태로 암묵적으로 승격/감쇠된다.
        list->CopyBufferRegion(buffer.Get(), 0, staging.Get(), 0, size);
        HR(list->Close());
        ID3D12CommandList* lists[] = {list.Get()};
        queue->ExecuteCommandLists(1, lists);
        waitIdle();
        return buffer;
    }

    void createPipeline() {
        D3D12_ROOT_PARAMETER params[2] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; // b0: 프레임 상수
        params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; // t0: 인스턴스 버퍼 (드로우마다 주소만 바꿈)
        params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;

        D3D12_ROOT_SIGNATURE_DESC rs{};
        rs.NumParameters = 2;
        rs.pParameters = params;
        rs.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
                   D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
                   D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
                   D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS;

        ComPtr<ID3DBlob> blob, error;
        if (FAILED(D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error))) {
            MEALKIT_FATAL("Root signature: %s",
                        error ? static_cast<const char*>(error->GetBufferPointer()) : "unknown error");
        }
        HR(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                       IID_PPV_ARGS(&rootSignature)));

        const D3D12_INPUT_ELEMENT_DESC layout[] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        };

        D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = rootSignature.Get();
        pd.VS = {g_ForwardVS, sizeof(g_ForwardVS)};
        pd.PS = {g_ForwardPS, sizeof(g_ForwardPS)};
        pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pd.SampleMask = UINT_MAX;
        pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pd.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
        pd.RasterizerState.DepthClipEnable = TRUE;
        pd.DepthStencilState.DepthEnable = TRUE;
        pd.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        pd.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_GREATER; // reversed-Z
        pd.InputLayout = {layout, static_cast<UINT>(std::size(layout))};
        pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pd.NumRenderTargets = 1;
        pd.RTVFormats[0] = kRtvFormat;
        pd.DSVFormat = kDepthFormat;
        pd.SampleDesc.Count = 1;
        HR(device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pipeline)));
    }

    void createSwapchainResources() {
        D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtvHeap->GetCPUDescriptorHandleForHeapStart();
        for (u32 i = 0; i < kBackBufferCount; ++i) {
            HR(swapchain->GetBuffer(i, IID_PPV_ARGS(&backBuffers[i])));
            D3D12_RENDER_TARGET_VIEW_DESC rd{};
            rd.Format = kRtvFormat;
            rd.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
            device->CreateRenderTargetView(backBuffers[i].Get(), &rd, {rtv.ptr + i * rtvStride});
            rd.Format = kOverlayFormat;
            device->CreateRenderTargetView(backBuffers[i].Get(), &rd, {rtv.ptr + (kBackBufferCount + i) * rtvStride});
        }

        D3D12_RESOURCE_DESC dd{};
        dd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        dd.Width = width;
        dd.Height = height;
        dd.DepthOrArraySize = 1;
        dd.MipLevels = 1;
        dd.Format = kDepthFormat;
        dd.SampleDesc.Count = 1;
        dd.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
        D3D12_CLEAR_VALUE clear{};
        clear.Format = kDepthFormat;
        clear.DepthStencil.Depth = 0.0f;
        const D3D12_HEAP_PROPERTIES defaultHeap = heapProperties(D3D12_HEAP_TYPE_DEFAULT);
        HR(device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &dd, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                           &clear, IID_PPV_ARGS(&depthBuffer)));
        D3D12_DEPTH_STENCIL_VIEW_DESC dsv{};
        dsv.Format = kDepthFormat;
        dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        device->CreateDepthStencilView(depthBuffer.Get(), &dsv, dsvHeap->GetCPUDescriptorHandleForHeapStart());
    }

    void releaseSwapchainResources() {
        for (auto& bb : backBuffers) bb.Reset();
        depthBuffer.Reset();
    }

    void savePng(ID3D12Resource* readback, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& fp, u64 totalBytes) {
        const u32 w = fp.Footprint.Width, h = fp.Footprint.Height;
        std::vector<u8> pixels(static_cast<size_t>(w) * h * 4);
        u8* data = nullptr;
        const D3D12_RANGE readRange{0, static_cast<SIZE_T>(totalBytes)};
        if (FAILED(readback->Map(0, &readRange, reinterpret_cast<void**>(&data)))) return;
        for (u32 y = 0; y < h; ++y) {
            const u8* src = data + fp.Offset + static_cast<u64>(y) * fp.Footprint.RowPitch;
            u8* dst = pixels.data() + static_cast<size_t>(y) * w * 4;
            for (u32 x = 0; x < w; ++x) { // RGBA -> BGRA
                dst[x * 4 + 0] = src[x * 4 + 2];
                dst[x * 4 + 1] = src[x * 4 + 1];
                dst[x * 4 + 2] = src[x * 4 + 0];
                dst[x * 4 + 3] = 255;
            }
        }
        const D3D12_RANGE noWrite{0, 0};
        readback->Unmap(0, &noWrite);

        const HRESULT co = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        bool ok = false;
        {
            ComPtr<IWICImagingFactory> wic;
            ComPtr<IWICStream> stream;
            ComPtr<IWICBitmapEncoder> encoder;
            ComPtr<IWICBitmapFrameEncode> frame;
            WICPixelFormatGUID format = GUID_WICPixelFormat32bppBGRA;
            const std::wstring path = widen(capturePath.c_str());
            ok = SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&wic))) &&
                 SUCCEEDED(wic->CreateStream(&stream)) &&
                 SUCCEEDED(stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE)) &&
                 SUCCEEDED(wic->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder)) &&
                 SUCCEEDED(encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache)) &&
                 SUCCEEDED(encoder->CreateNewFrame(&frame, nullptr)) && SUCCEEDED(frame->Initialize(nullptr)) &&
                 SUCCEEDED(frame->SetSize(w, h)) && SUCCEEDED(frame->SetPixelFormat(&format)) &&
                 format == GUID_WICPixelFormat32bppBGRA &&
                 SUCCEEDED(frame->WritePixels(h, w * 4, static_cast<UINT>(pixels.size()), pixels.data())) &&
                 SUCCEEDED(frame->Commit()) && SUCCEEDED(encoder->Commit());
        }
        if (SUCCEEDED(co)) CoUninitialize();
        if (ok) MEALKIT_INFO("Screenshot saved: %s (%ux%u)", capturePath.c_str(), w, h);
        else MEALKIT_WARN("Screenshot failed: %s", capturePath.c_str());
    }
};

Renderer::~Renderer() { shutdown(); }

bool Renderer::init(const RendererDesc& desc) {
    m = new Impl();
    m->hwnd = static_cast<HWND>(desc.window);
    m->width = desc.width;
    m->height = desc.height;
    m->vsync = desc.vsync;

    UINT factoryFlags = 0;
    if (desc.debugLayer) {
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
            debug->EnableDebugLayer();
            factoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
            MEALKIT_INFO("D3D12 debug layer enabled");
        } else {
            MEALKIT_WARN("D3D12 debug layer unavailable (install the 'Graphics Tools' optional feature)");
        }
    }

    if (FAILED(CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&m->factory)))) {
        MEALKIT_ERROR("DXGI 1.6 is required (Windows 10 1803 or later)");
        return false;
    }
    if (!m->pickAdapter()) {
        MEALKIT_ERROR("No Direct3D 12 capable GPU found");
        return false;
    }
    HR(D3D12CreateDevice(m->adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m->device)));
    g_errorDevice = m->device.Get();

    const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_12_2, D3D_FEATURE_LEVEL_12_1, D3D_FEATURE_LEVEL_12_0,
                                        D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
    D3D12_FEATURE_DATA_FEATURE_LEVELS fl{_countof(levels), levels, D3D_FEATURE_LEVEL_11_0};
    m->device->CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &fl, sizeof(fl));

    D3D12_FEATURE_DATA_SHADER_MODEL sm{D3D_SHADER_MODEL_6_0};
    if (FAILED(m->device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm))) ||
        sm.HighestShaderModel < D3D_SHADER_MODEL_6_0) {
        MEALKIT_ERROR("Shader Model 6.0 is required (update the GPU driver)");
        return false;
    }

    D3D12_FEATURE_DATA_ARCHITECTURE arch{};
    m->device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &arch, sizeof(arch));
    m->uma = arch.UMA != FALSE;
    MEALKIT_INFO("GPU: %s | feature level %X | %s memory", m->adapterName.c_str(),
               static_cast<unsigned>(fl.MaxSupportedFeatureLevel), m->uma ? "unified (UMA)" : "dedicated");

    if (desc.debugLayer) {
        ComPtr<ID3D12InfoQueue1> infoQueue;
        if (SUCCEEDED(m->device.As(&infoQueue))) {
            DWORD cookie = 0;
            infoQueue->RegisterMessageCallback(onDebugMessage, D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr, &cookie);
        }
    }

    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    HR(m->device->CreateCommandQueue(&qd, IID_PPV_ARGS(&m->queue)));

    BOOL allowTearing = FALSE;
    if (SUCCEEDED(m->factory->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allowTearing,
                                                  sizeof(allowTearing))))
        m->tearingSupported = allowTearing != FALSE;

    m->swapchainFlags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT |
                        (m->tearingSupported ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0u);
    DXGI_SWAP_CHAIN_DESC1 sd{};
    sd.Width = m->width;
    sd.Height = m->height;
    sd.Format = kSwapchainFormat;
    sd.SampleDesc.Count = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = kBackBufferCount;
    sd.Scaling = DXGI_SCALING_STRETCH;
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    sd.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
    sd.Flags = m->swapchainFlags;
    ComPtr<IDXGISwapChain1> swapchain1;
    HR(m->factory->CreateSwapChainForHwnd(m->queue.Get(), m->hwnd, &sd, nullptr, nullptr, &swapchain1));
    HR(m->factory->MakeWindowAssociation(m->hwnd, DXGI_MWA_NO_ALT_ENTER));
    HR(swapchain1.As(&m->swapchain));
    HR(m->swapchain->SetMaximumFrameLatency(kFramesInFlight));
    m->frameLatencyWaitable = m->swapchain->GetFrameLatencyWaitableObject();

    D3D12_DESCRIPTOR_HEAP_DESC rtvDesc{};
    rtvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvDesc.NumDescriptors = kBackBufferCount * 2; // sRGB 뷰 + 오버레이 뷰
    HR(m->device->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(&m->rtvHeap)));
    D3D12_DESCRIPTOR_HEAP_DESC dsvDesc{};
    dsvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvDesc.NumDescriptors = 1;
    HR(m->device->CreateDescriptorHeap(&dsvDesc, IID_PPV_ARGS(&m->dsvHeap)));
    m->rtvStride = m->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    for (auto& frame : m->frames)
        HR(m->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&frame.allocator)));
    HR(m->device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m->frames[0].allocator.Get(), nullptr,
                                    IID_PPV_ARGS(&m->cmd)));
    HR(m->cmd->Close());

    HR(m->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m->fence)));
    m->fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    const D3D12_HEAP_PROPERTIES uploadHeap = heapProperties(D3D12_HEAP_TYPE_UPLOAD);
    const D3D12_RESOURCE_DESC uploadDesc = bufferDesc(kUploadBytesPerFrame * kFramesInFlight);
    HR(m->device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &uploadDesc,
                                          D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m->uploadBuffer)));
    const D3D12_RANGE noRead{0, 0};
    HR(m->uploadBuffer->Map(0, &noRead, reinterpret_cast<void**>(&m->uploadCpu)));
    m->uploadGpu = m->uploadBuffer->GetGPUVirtualAddress();

    m->createPipeline();
    m->createSwapchainResources();
#if defined(TRACY_ENABLE)
    m->tracyGpu = TracyD3D12Context(m->device.Get(), m->queue.Get());
#endif
    return true;
}

void Renderer::shutdown() {
    if (!m) return;
    if (m->queue && m->fence) m->waitIdle();
#if defined(TRACY_ENABLE)
    if (m->tracyGpu) TracyD3D12Destroy(m->tracyGpu);
#endif
    if (m->fenceEvent) CloseHandle(m->fenceEvent);
    if (m->frameLatencyWaitable) CloseHandle(m->frameLatencyWaitable);
    delete m;
    m = nullptr;
    g_errorDevice = nullptr;
}

void Renderer::resize(u32 width, u32 height) {
    if (!m || width == 0 || height == 0 || (width == m->width && height == m->height)) return;
    m->waitIdle();
    m->releaseSwapchainResources();
    HR(m->swapchain->ResizeBuffers(kBackBufferCount, width, height, DXGI_FORMAT_UNKNOWN, m->swapchainFlags));
    m->width = width;
    m->height = height;
    m->createSwapchainResources();
}

MeshId Renderer::createMesh(const MeshData& data) {
    if (data.vertices.empty() || data.indices.empty()) {
        MEALKIT_WARN("createMesh: empty mesh");
        return 0;
    }
    GpuMesh mesh;
    const u64 vbBytes = data.vertices.size() * sizeof(Vertex);
    mesh.vertexBuffer = m->createStaticBuffer(data.vertices.data(), vbBytes);
    mesh.vbv = {mesh.vertexBuffer->GetGPUVirtualAddress(), static_cast<UINT>(vbBytes), sizeof(Vertex)};

    // 정점이 적으면 16비트 인덱스로 줄여 메모리 대역폭을 아낀다 (내장 그래픽에서 중요)
    u64 ibBytes = 0;
    DXGI_FORMAT indexFormat = DXGI_FORMAT_R32_UINT;
    if (data.vertices.size() <= 0xFFFF) {
        std::vector<u16> shortIndices(data.indices.begin(), data.indices.end());
        ibBytes = shortIndices.size() * sizeof(u16);
        mesh.indexBuffer = m->createStaticBuffer(shortIndices.data(), ibBytes);
        indexFormat = DXGI_FORMAT_R16_UINT;
    } else {
        ibBytes = data.indices.size() * sizeof(u32);
        mesh.indexBuffer = m->createStaticBuffer(data.indices.data(), ibBytes);
    }
    mesh.ibv = {mesh.indexBuffer->GetGPUVirtualAddress(), static_cast<UINT>(ibBytes), indexFormat};
    mesh.indexCount = static_cast<u32>(data.indices.size());
    m->meshes.push_back(std::move(mesh));
    return static_cast<MeshId>(m->meshes.size() - 1);
}

bool Renderer::beginFrame(const FrameParams& p) {
    if (!m || m->width == 0 || m->height == 0) return false;

    // 프레젠트 큐가 비기를 기다린다: CPU가 GPU보다 너무 앞서가 입력 지연이 쌓이는 것을 막는다.
    WaitForSingleObjectEx(m->frameLatencyWaitable, 1000, TRUE);

    Impl::Frame& frame = m->frames[m->frameIndex];
    m->waitForFence(frame.fenceValue);
    HR(frame.allocator->Reset());
    HR(m->cmd->Reset(frame.allocator.Get(), m->pipeline.Get()));
#if defined(TRACY_ENABLE)
    static const tracy::SourceLocationData gpuFrameLocation{"GPU frame", __FUNCTION__, __FILE__, __LINE__, 0};
    if (m->tracyGpu) m->tracyFrameZone.emplace(m->tracyGpu, m->cmd.Get(), &gpuFrameLocation, true);
#endif

    m->uploadOffset = m->frameIndex * kUploadBytesPerFrame;
    m->uploadLimit = m->uploadOffset + kUploadBytesPerFrame;
    m->backBufferIndex = m->swapchain->GetCurrentBackBufferIndex();

    ID3D12GraphicsCommandList* cmd = m->cmd.Get();
    ID3D12Resource* backBuffer = m->backBuffers[m->backBufferIndex].Get();
    const D3D12_RESOURCE_BARRIER toRender =
        transition(backBuffer, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
    cmd->ResourceBarrier(1, &toRender);

    const D3D12_CPU_DESCRIPTOR_HANDLE rtv{m->rtvHeap->GetCPUDescriptorHandleForHeapStart().ptr +
                                          m->backBufferIndex * m->rtvStride};
    const D3D12_CPU_DESCRIPTOR_HANDLE dsv = m->dsvHeap->GetCPUDescriptorHandleForHeapStart();
    cmd->OMSetRenderTargets(1, &rtv, FALSE, &dsv);
    const f32 clear[4] = {p.fogColor.x, p.fogColor.y, p.fogColor.z, 1.0f};
    cmd->ClearRenderTargetView(rtv, clear, 0, nullptr);
    cmd->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 0.0f, 0, 0, nullptr);

    const f32 fullW = static_cast<f32>(m->width), fullH = static_cast<f32>(m->height);
    f32 vx = 0.0f, vy = 0.0f, vw = fullW, vh = fullH;
    if (p.viewport.width >= 1.0f && p.viewport.height >= 1.0f) {
        vx = std::clamp(p.viewport.x, 0.0f, fullW - 1.0f);
        vy = std::clamp(p.viewport.y, 0.0f, fullH - 1.0f);
        vw = std::min(p.viewport.width, fullW - vx);
        vh = std::min(p.viewport.height, fullH - vy);
    }
    const D3D12_VIEWPORT viewport{vx, vy, vw, vh, 0.0f, 1.0f};
    const D3D12_RECT scissor{static_cast<LONG>(vx), static_cast<LONG>(vy), static_cast<LONG>(vx + vw),
                             static_cast<LONG>(vy + vh)};
    cmd->RSSetViewports(1, &viewport);
    cmd->RSSetScissorRects(1, &scissor);
    cmd->SetGraphicsRootSignature(m->rootSignature.Get());
    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    FrameConstants fc{};
    std::memcpy(fc.viewProj, p.viewProj.m, sizeof(fc.viewProj));
    const Vec3 sun = normalize(p.sunDirection);
    const f32 cameraPos[4] = {p.cameraPos.x, p.cameraPos.y, p.cameraPos.z, 0.0f};
    const f32 sunDir[4] = {sun.x, sun.y, sun.z, 0.0f};
    const f32 sunColor[4] = {p.sunColor.x, p.sunColor.y, p.sunColor.z, 0.0f};
    const f32 skyColor[4] = {p.skyColor.x, p.skyColor.y, p.skyColor.z, 0.0f};
    const f32 groundColor[4] = {p.groundColor.x, p.groundColor.y, p.groundColor.z, 0.0f};
    const f32 fogColor[4] = {p.fogColor.x, p.fogColor.y, p.fogColor.z, p.fogDensity};
    std::memcpy(fc.cameraPos, cameraPos, sizeof(cameraPos));
    std::memcpy(fc.sunDir, sunDir, sizeof(sunDir));
    std::memcpy(fc.sunColor, sunColor, sizeof(sunColor));
    std::memcpy(fc.skyColor, skyColor, sizeof(skyColor));
    std::memcpy(fc.groundColor, groundColor, sizeof(groundColor));
    std::memcpy(fc.fogColor, fogColor, sizeof(fogColor));

    void* cpu = nullptr;
    const D3D12_GPU_VIRTUAL_ADDRESS gpu = m->allocUpload(sizeof(fc), &cpu);
    std::memcpy(cpu, &fc, sizeof(fc));
    cmd->SetGraphicsRootConstantBufferView(0, gpu);

    m->stats = {};
    m->inFrame = true;
    return true;
}

void Renderer::drawInstances(MeshId meshId, const InstanceData* instances, u32 count) {
    if (!m || !m->inFrame || count == 0 || meshId >= m->meshes.size()) return;

    const u64 bytes = static_cast<u64>(count) * sizeof(InstanceData);
    void* cpu = nullptr;
    const D3D12_GPU_VIRTUAL_ADDRESS gpu = m->allocUpload(bytes, &cpu);
    if (!gpu) {
        if (!m->uploadOverflowWarned) {
            MEALKIT_WARN("Upload ring full (%llu bytes requested); raise kUploadBytesPerFrame",
                       static_cast<unsigned long long>(bytes));
            m->uploadOverflowWarned = true;
        }
        return;
    }
    std::memcpy(cpu, instances, bytes);

    const GpuMesh& mesh = m->meshes[meshId];
    ID3D12GraphicsCommandList* cmd = m->cmd.Get();
    cmd->SetGraphicsRootShaderResourceView(1, gpu);
    cmd->IASetVertexBuffers(0, 1, &mesh.vbv);
    cmd->IASetIndexBuffer(&mesh.ibv);
    cmd->DrawIndexedInstanced(mesh.indexCount, count, 0, 0, 0);
    m->stats.drawCalls += 1;
    m->stats.instances += count;
}

void Renderer::beginOverlay() {
    if (!m || !m->inFrame) return;
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv{m->rtvHeap->GetCPUDescriptorHandleForHeapStart().ptr +
                                          (kBackBufferCount + m->backBufferIndex) * m->rtvStride};
    m->cmd->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
}

void Renderer::waitIdle() {
    if (m && m->queue && m->fence) m->waitIdle();
}

NativeHandles Renderer::native() const {
    NativeHandles h;
    if (!m) return h;
    h.device = m->device.Get();
    h.queue = m->queue.Get();
    h.commandList = m->inFrame ? m->cmd.Get() : nullptr;
    h.framesInFlight = kFramesInFlight;
    h.overlayFormat = static_cast<u32>(kOverlayFormat);
    return h;
}

u32 Renderer::width() const { return m ? m->width : 0; }
u32 Renderer::height() const { return m ? m->height : 0; }

void Renderer::endFrame() {
    if (!m || !m->inFrame) return;
    ID3D12GraphicsCommandList* cmd = m->cmd.Get();
    ID3D12Resource* backBuffer = m->backBuffers[m->backBufferIndex].Get();

    ComPtr<ID3D12Resource> readback;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    u64 readbackBytes = 0;
    if (!m->capturePath.empty()) {
        const D3D12_RESOURCE_DESC texDesc = backBuffer->GetDesc();
        m->device->GetCopyableFootprints(&texDesc, 0, 1, 0, &footprint, nullptr, nullptr, &readbackBytes);
        const D3D12_HEAP_PROPERTIES readbackHeap = heapProperties(D3D12_HEAP_TYPE_READBACK);
        const D3D12_RESOURCE_DESC rbDesc = bufferDesc(readbackBytes);
        HR(m->device->CreateCommittedResource(&readbackHeap, D3D12_HEAP_FLAG_NONE, &rbDesc,
                                              D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback)));
        const D3D12_RESOURCE_BARRIER toCopy =
            transition(backBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        cmd->ResourceBarrier(1, &toCopy);
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = readback.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = footprint;
        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = backBuffer;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = 0;
        cmd->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        const D3D12_RESOURCE_BARRIER toPresent =
            transition(backBuffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT);
        cmd->ResourceBarrier(1, &toPresent);
    } else {
        const D3D12_RESOURCE_BARRIER toPresent =
            transition(backBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
        cmd->ResourceBarrier(1, &toPresent);
    }

#if defined(TRACY_ENABLE)
    m->tracyFrameZone.reset(); // GPU 구간 끝 (커맨드 리스트를 닫기 전에)
#endif
    HR(cmd->Close());
    ID3D12CommandList* lists[] = {cmd};
    m->queue->ExecuteCommandLists(1, lists);

    const UINT syncInterval = m->vsync ? 1 : 0;
    const UINT presentFlags = (!m->vsync && m->tearingSupported) ? DXGI_PRESENT_ALLOW_TEARING : 0;
    HR(m->swapchain->Present(syncInterval, presentFlags));

    Impl::Frame& frame = m->frames[m->frameIndex];
    frame.fenceValue = m->signal();
#if defined(TRACY_ENABLE)
    if (m->tracyGpu) {
        TracyD3D12NewFrame(m->tracyGpu);
        TracyD3D12Collect(m->tracyGpu);
    }
#endif

    if (readback) {
        m->waitForFence(frame.fenceValue);
        m->savePng(readback.Get(), footprint, readbackBytes);
        m->capturePath.clear();
    }

    m->frameIndex = (m->frameIndex + 1) % kFramesInFlight;
    m->inFrame = false;
}

void Renderer::setVsync(bool enabled) {
    if (m) m->vsync = enabled;
}
bool Renderer::vsync() const { return m && m->vsync; }
void Renderer::requestCapture(const char* pngPath) {
    if (m) m->capturePath = pngPath;
}
const char* Renderer::adapterName() const { return m ? m->adapterName.c_str() : ""; }
bool Renderer::isUma() const { return m && m->uma; }
const RenderStats& Renderer::stats() const {
    static const RenderStats empty{};
    return m ? m->stats : empty;
}

} // namespace mealkit::render
