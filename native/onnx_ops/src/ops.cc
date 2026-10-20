#include "onnxruntime_c_api.h"
#include "kernels.h"
#include <cudnn.h>
#include <cudnn_frontend.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#if defined(_WIN32)
#define OPS_EXPORT extern "C" __declspec(dllexport)
#else
#define OPS_EXPORT extern "C" __attribute__((visibility("default")))
#endif
namespace {
namespace fe = cudnn_frontend;
using Dims = std::vector<int64_t>;
std::atomic<uint64_t> builds{0}, hits{0}, evictions{0};
std::atomic<uint64_t> graph_ns{0}, heuristics_ns{0}, support_ns{0}, plan_ns{0}, classic_builds{0}, classic_fallbacks{0};
template<typename F> void Timed(std::atomic<uint64_t>& counter, F&& f) {
  auto start=std::chrono::steady_clock::now(); f();
  counter += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
}
void Cuda(cudaError_t s) { if(s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
void Cudnn(cudnnStatus_t s) { if(s != CUDNN_STATUS_SUCCESS) throw std::runtime_error(cudnnGetErrorString(s)); }
void Frontend(fe::error_t s) { if(s.is_bad()) throw std::runtime_error(s.get_message()); }
struct Api {
  const OrtApi* a;
  void Check(OrtStatus* s) const { if(s) { std::string msg=a->GetErrorMessage(s); a->ReleaseStatus(s); throw std::runtime_error(msg); } }
  int64_t Integer(const OrtKernelInfo* info,const char* key,int64_t fallback) const {
    int64_t value; OrtStatus* s=a->KernelInfoGetAttribute_int64(info,key,&value);
    if(s) { a->ReleaseStatus(s); return fallback; } return value;
  }
  Dims Array(const OrtKernelInfo* info,const char* key,Dims fallback) const {
    size_t n=0; OrtStatus* s=a->KernelInfoGetAttributeArray_int64(info,key,nullptr,&n);
    if(s) { a->ReleaseStatus(s); return fallback; }
    Dims out(n); Check(a->KernelInfoGetAttributeArray_int64(info,key,out.data(),&n)); return out;
  }
  std::string String(const OrtKernelInfo* info,const char* key,std::string fallback) const {
    size_t n=0;OrtStatus* s=a->KernelInfoGetAttribute_string(info,key,nullptr,&n);
    if(s){a->ReleaseStatus(s);return fallback;}
    std::vector<char> out(n);Check(a->KernelInfoGetAttribute_string(info,key,out.data(),&n));return std::string(out.data());
  }
  Dims Shape(const OrtValue* v) const {
    OrtTensorTypeAndShapeInfo* info=nullptr; Check(a->GetTensorTypeAndShape(v,&info));
    struct Release { const OrtApi* a; OrtTensorTypeAndShapeInfo* p; ~Release(){a->ReleaseTensorTypeAndShapeInfo(p);} } guard{a,info};
    size_t rank; Check(a->GetDimensionsCount(info,&rank)); Dims out(rank); Check(a->GetDimensions(info,out.data(),rank)); return out;
  }
  const OrtValue* Input(OrtKernelContext* ctx,size_t i) const { const OrtValue* v=nullptr; Check(a->KernelContext_GetInput(ctx,i,&v)); return v; }
  float* Data(OrtValue* v) const { void* p=nullptr; Check(a->GetTensorMutableData(v,&p)); return static_cast<float*>(p); }
  void* Output(OrtKernelContext* ctx,size_t i,const Dims& shape) const {
    OrtValue* v=nullptr; Check(a->KernelContext_GetOutput(ctx,i,shape.data(),shape.size(),&v)); return Data(v);
  }
  cudaStream_t Stream(OrtKernelContext* ctx) const { void* p=nullptr; Check(a->KernelContext_GetGPUComputeStream(ctx,&p)); return static_cast<cudaStream_t>(p); }
};
Dims Strides(const Dims& shape) { Dims s(shape.size(),1); for(size_t i=shape.size()-1;i>0;--i)s[i-1]=s[i]*shape[i];return s; }
struct Workspace {
  std::mutex mutex; cudnnHandle_t handle=nullptr; void* data=nullptr; size_t bytes=0; int device;
  explicit Workspace(int d):device(d) { Cudnn(cudnnCreate(&handle)); }
  ~Workspace(){ int old=0;cudaGetDevice(&old);cudaSetDevice(device);if(data)cudaFree(data);if(handle)cudnnDestroy(handle);cudaSetDevice(old); }
  void Reserve(size_t n) { if(n>bytes){ void* p=nullptr;Cuda(cudaMalloc(&p,n));if(data)Cuda(cudaFree(data));data=p;bytes=n;} }
};
std::mutex workspace_mutex;
std::map<std::pair<int,uintptr_t>,std::weak_ptr<Workspace>> workspaces;
std::shared_ptr<Workspace> GetWorkspace(int device,cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(workspace_mutex);
  auto key=std::make_pair(device,reinterpret_cast<uintptr_t>(stream));auto found=workspaces.find(key);
  if(found!=workspaces.end())if(auto p=found->second.lock())return p;
  for(auto i=workspaces.begin();i!=workspaces.end();)if(i->second.expired())i=workspaces.erase(i);else ++i;
  auto p=std::make_shared<Workspace>(device);workspaces[key]=p;return p;
}
struct Plan {
  fe::graph::Graph graph;
  std::shared_ptr<fe::graph::Tensor_attributes> x,w,y;
  cudnnTensorDescriptor_t y_desc=nullptr,b_desc=nullptr;
  bool legacy=false;
  cudnnTensorDescriptor_t x_desc=nullptr;
  cudnnFilterDescriptor_t w_desc=nullptr;
  cudnnConvolutionDescriptor_t conv_desc=nullptr;
  cudnnConvolutionFwdAlgo_t algorithm=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  size_t workspace=0;
  ~Plan(){if(x_desc)cudnnDestroyTensorDescriptor(x_desc);if(w_desc)cudnnDestroyFilterDescriptor(w_desc);if(conv_desc)cudnnDestroyConvolutionDescriptor(conv_desc);if(y_desc)cudnnDestroyTensorDescriptor(y_desc);if(b_desc)cudnnDestroyTensorDescriptor(b_desc);}
};
struct Conv {
  Api api; Dims pads,strides,dilations; int64_t group; size_t capacity; bool tf32,classic; std::string auto_pad;
  std::mutex mutex;
  std::list<std::pair<Dims,std::shared_ptr<Plan>>> cache;
  std::map<std::pair<int,uintptr_t>,std::shared_ptr<Workspace>> scratch;
  Conv(const OrtApi* a,const OrtKernelInfo* info):api{a},pads(api.Array(info,"pads",{0,0,0,0})),strides(api.Array(info,"strides",{1,1})),dilations(api.Array(info,"dilations",{1,1})),group(api.Integer(info,"group",1)),capacity(static_cast<size_t>(api.Integer(info,"cache_capacity",64))),tf32(api.Integer(info,"use_tf32",1)!=0) {
    auto_pad=api.String(info,"auto_pad","NOTSET");
    auto conv_api=api.String(info,"conv_api","frontend");
    if(conv_api!="frontend"&&conv_api!="classic")throw std::runtime_error("Invalid conv_api");
    classic=conv_api=="classic";
    if(auto_pad!="NOTSET"&&auto_pad!="VALID"&&auto_pad!="SAME_UPPER"&&auto_pad!="SAME_LOWER")throw std::runtime_error("Invalid auto_pad");
    if(pads.size()!=4||strides.size()!=2||dilations.size()!=2||group<=0||capacity>128)throw std::runtime_error("Invalid Conv2D attributes");
    for(auto v:pads)if(v<0)throw std::runtime_error("Negative Conv padding unsupported");
    for(auto v:strides)if(v<=0)throw std::runtime_error("Invalid Conv stride");
    for(auto v:dilations)if(v<=0)throw std::runtime_error("Invalid Conv dilation");
  }
  std::shared_ptr<Plan> Build(const Dims& x,const Dims& w,const Dims& y,const Dims& padding,bool bias,cudnnHandle_t handle) {
    auto p=std::make_shared<Plan>();
    // Classic cuDNN avoids graph heuristics overhead for ordinary 2D Conv.
    // Asymmetric SAME padding and dimensions beyond descriptor limits use frontend.
    auto fits=[](const Dims& dims){return std::all_of(dims.begin(),dims.end(),[](int64_t v){return v>0&&v<=INT_MAX;});};
    if(classic && padding[0]==padding[2] && padding[1]==padding[3] && fits(x) && fits(w) && fits(y)
       && fits(strides) && fits(dilations) && padding[0]<=INT_MAX && padding[1]<=INT_MAX && group<=INT_MAX) {
      try {
      Timed(heuristics_ns,[&]{
        Cudnn(cudnnCreateTensorDescriptor(&p->x_desc));
        Cudnn(cudnnCreateTensorDescriptor(&p->y_desc));
        Cudnn(cudnnCreateFilterDescriptor(&p->w_desc));
        Cudnn(cudnnCreateConvolutionDescriptor(&p->conv_desc));
        Cudnn(cudnnSetTensor4dDescriptor(p->x_desc,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,x[0],x[1],x[2],x[3]));
        Cudnn(cudnnSetTensor4dDescriptor(p->y_desc,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,y[0],y[1],y[2],y[3]));
        Cudnn(cudnnSetFilter4dDescriptor(p->w_desc,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,w[0],w[1],w[2],w[3]));
        Cudnn(cudnnSetConvolution2dDescriptor(p->conv_desc,padding[0],padding[1],strides[0],strides[1],dilations[0],dilations[1],CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT));
        Cudnn(cudnnSetConvolutionGroupCount(p->conv_desc,group));
        Cudnn(cudnnSetConvolutionMathType(p->conv_desc,tf32?CUDNN_DEFAULT_MATH:CUDNN_FMA_MATH));
        cudnnConvolutionFwdAlgoPerf_t perf[8]; int count=0;
        Cudnn(cudnnGetConvolutionForwardAlgorithm_v7(handle,p->x_desc,p->w_desc,p->conv_desc,p->y_desc,8,&count,perf));
        for(int i=0;i<count;++i) {
          if(perf[i].status!=CUDNN_STATUS_SUCCESS || (!tf32 && (perf[i].mathType==CUDNN_TENSOR_OP_MATH || perf[i].mathType==CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)))continue;
          Cudnn(cudnnSetConvolutionMathType(p->conv_desc,tf32?perf[i].mathType:CUDNN_FMA_MATH));
          auto s=cudnnGetConvolutionForwardWorkspaceSize(handle,p->x_desc,p->w_desc,p->conv_desc,p->y_desc,perf[i].algo,&p->workspace);
          if(s==CUDNN_STATUS_SUCCESS){p->algorithm=perf[i].algo;p->legacy=true;break;}
        }
        if(!p->legacy)throw std::runtime_error("No legacy convolution algorithm");
      });
      if(bias){Cudnn(cudnnCreateTensorDescriptor(&p->b_desc));Cudnn(cudnnSetTensor4dDescriptor(p->b_desc,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,1,y[1],1,1));}
      ++builds;++classic_builds;return p;
      } catch(const std::runtime_error&) {
        // Fall back only during plan preparation; never retry failed GPU execution.
        ++classic_fallbacks;p=std::make_shared<Plan>();
      }
    }
    p->graph.set_io_data_type(fe::DataType_t::FLOAT).set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
    p->x=p->graph.tensor(fe::graph::Tensor_attributes().set_name("X").set_dim(x).set_stride(Strides(x)));
    p->w=p->graph.tensor(fe::graph::Tensor_attributes().set_name("W").set_dim(w).set_stride(Strides(w)));
    p->y=p->graph.conv_fprop(p->x,p->w,fe::graph::Conv_fprop_attributes().set_pre_padding({padding[0],padding[1]}).set_post_padding({padding[2],padding[3]}).set_stride(strides).set_dilation(dilations));
    p->y->set_output(true).set_dim(y).set_stride(Strides(y));
    Frontend(p->graph.validate());Timed(graph_ns,[&]{Frontend(p->graph.build_operation_graph(handle));});Timed(heuristics_ns,[&]{Frontend(p->graph.create_execution_plans({fe::HeurMode_t::B}));});
    if(!tf32)p->graph.deselect_numeric_notes({fe::NumericalNote_t::TENSOR_CORE});
    Timed(support_ns,[&]{Frontend(p->graph.check_support(handle));});Timed(plan_ns,[&]{Frontend(p->graph.build_plans(handle));});p->workspace=p->graph.get_workspace_size();
    if(bias) {
      for(auto v:y)if(v>INT_MAX)throw std::runtime_error("Conv bias tensor dimension exceeds int32");
      Cudnn(cudnnCreateTensorDescriptor(&p->y_desc));Cudnn(cudnnCreateTensorDescriptor(&p->b_desc));
      Cudnn(cudnnSetTensor4dDescriptor(p->y_desc,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,static_cast<int>(y[0]),static_cast<int>(y[1]),static_cast<int>(y[2]),static_cast<int>(y[3])));
      Cudnn(cudnnSetTensor4dDescriptor(p->b_desc,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,1,static_cast<int>(y[1]),1,1));
    }
    ++builds;return p;
  }
  void Compute(OrtKernelContext* ctx) {
    std::lock_guard<std::mutex> guard(mutex);
    auto xv=api.Input(ctx,0),wv=api.Input(ctx,1);size_t n=0;api.Check(api.a->KernelContext_GetInputCount(ctx,&n));const OrtValue* bv=n>2?api.Input(ctx,2):nullptr;
    Dims x=api.Shape(xv),w=api.Shape(wv);
    if(x.size()!=4||w.size()!=4||x[1]!=w[1]*group||w[0]%group)throw std::runtime_error("Conv2D expects valid NCHW grouped convolution");
    for(auto d:w)if(d<=0)throw std::runtime_error("Conv weights must have positive dimensions");
    for(size_t i=1;i<x.size();++i)if(x[i]<=0)throw std::runtime_error("Conv input requires positive channel/spatial dimensions");
    Dims padding=pads;
    if(auto_pad=="VALID")padding={0,0,0,0};
    if(auto_pad=="SAME_UPPER"||auto_pad=="SAME_LOWER")for(size_t i=0;i<2;++i){
      auto output=(x[i+2]+strides[i]-1)/strides[i];
      auto total=std::max<int64_t>(0,(output-1)*strides[i]+dilations[i]*(w[i+2]-1)+1-x[i+2]);
      padding[i]=auto_pad=="SAME_UPPER"?total/2:(total+1)/2;padding[i+2]=total-padding[i];
    }
    Dims y{x[0],w[0],0,0};for(size_t i=0;i<2;++i){int64_t extent=x[i+2]+padding[i]+padding[i+2]-dilations[i]*(w[i+2]-1)-1;if(extent<0)throw std::runtime_error("Negative Conv output extent");y[i+2]=extent/strides[i]+1;}
    if(bv&&api.Shape(bv)!=Dims{w[0]})throw std::runtime_error("Conv2D bias must be [output_channels]");
    auto out=static_cast<float*>(api.Output(ctx,0,y));if(y[0]==0)return;
    auto stream=api.Stream(ctx);int device;Cuda(cudaGetDevice(&device));auto key_stream=std::make_pair(device,reinterpret_cast<uintptr_t>(stream));auto& workspace=scratch[key_stream];if(!workspace)workspace=GetWorkspace(device,stream);
    std::lock_guard<std::mutex> lock(workspace->mutex);Cudnn(cudnnSetStream(workspace->handle,stream));
    Dims key=x;key.insert(key.end(),w.begin(),w.end());key.push_back(bv!=nullptr);key.push_back(device);
    auto it=std::find_if(cache.begin(),cache.end(),[&](const auto& v){return v.first==key;});std::shared_ptr<Plan> plan;
    if(it!=cache.end()){plan=it->second;cache.splice(cache.begin(),cache,it);++hits;}
    else {plan=Build(x,w,y,padding,bv!=nullptr,workspace->handle);if(capacity){cache.emplace_front(key,plan);if(cache.size()>capacity){cache.pop_back();++evictions;}}}
    workspace->Reserve(plan->workspace);
    if(plan->legacy) {
      const float alpha=1.f,beta=0.f;
      Cudnn(cudnnConvolutionForward(workspace->handle,&alpha,plan->x_desc,api.Data(const_cast<OrtValue*>(xv)),plan->w_desc,api.Data(const_cast<OrtValue*>(wv)),plan->conv_desc,plan->algorithm,workspace->data,plan->workspace,&beta,plan->y_desc,out));
    } else {
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>,void*> pack{{plan->x,api.Data(const_cast<OrtValue*>(xv))},{plan->w,api.Data(const_cast<OrtValue*>(wv))},{plan->y,out}};
    Frontend(plan->graph.execute(workspace->handle,pack,workspace->data));
    }
    if(bv){const float alpha=1.f;Cudnn(cudnnAddTensor(workspace->handle,&alpha,plan->b_desc,api.Data(const_cast<OrtValue*>(bv)),&alpha,plan->y_desc,out));}
  }
};
struct Max {
  Api api; bool keepdims,largest;
  Max(const OrtApi* a,const OrtKernelInfo* info):api{a},keepdims(api.Integer(info,"keepdims",0)!=0),largest(api.Integer(info,"largest",1)!=0){}
  void Compute(OrtKernelContext* ctx){ auto v=api.Input(ctx,0);auto shape=api.Shape(v);if(shape.empty()||shape.back()<=0)throw std::runtime_error("MaxWithIndex needs a nonempty last dimension");auto width=shape.back();shape.pop_back();int64_t rows=1;for(auto d:shape)rows*=d;if(keepdims)shape.push_back(1);
    auto values=static_cast<float*>(api.Output(ctx,0,shape));auto indices=static_cast<int64_t*>(api.Output(ctx,1,shape));RowMax(api.Stream(ctx),api.Data(const_cast<OrtValue*>(v)),values,indices,rows,width,largest);Cuda(cudaGetLastError()); }
};
template<typename F> OrtStatus* Guard(const OrtApi* api,F&& f){try{f();return nullptr;}catch(const std::exception& e){return api->CreateStatus(ORT_RUNTIME_EXCEPTION,e.what());}catch(...){return api->CreateStatus(ORT_RUNTIME_EXCEPTION,"Unknown onnxruntime-ops error");}}
template<typename K> OrtCustomOp MakeOp(const char* name,size_t inputs,size_t outputs,bool conv){
  OrtCustomOp op{};op.version=16;
  op.GetExecutionProviderType=[](const OrtCustomOp*){return "CUDAExecutionProvider";};
  op.GetInputType=[](const OrtCustomOp*,size_t){return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;};
  op.GetInputMemoryType=[](const OrtCustomOp*,size_t){return OrtMemTypeDefault;};
  op.GetInputCharacteristic=[](const OrtCustomOp* p,size_t i){return p->GetInputTypeCount(p)==3&&i==2?INPUT_OUTPUT_OPTIONAL:INPUT_OUTPUT_REQUIRED;};
  op.GetOutputCharacteristic=[](const OrtCustomOp*,size_t){return INPUT_OUTPUT_REQUIRED;};
  op.GetVariadicInputMinArity=[](const OrtCustomOp*){return 1;};op.GetVariadicOutputMinArity=op.GetVariadicInputMinArity;
  op.GetVariadicInputHomogeneity=[](const OrtCustomOp*){return 1;};op.GetVariadicOutputHomogeneity=op.GetVariadicInputHomogeneity;
  op.CreateKernelV2=[](const OrtCustomOp*,const OrtApi* a,const OrtKernelInfo* info,void** out){return Guard(a,[&]{*out=new K(a,info);});};
  op.KernelComputeV2=[](void* k,OrtKernelContext* ctx){auto p=static_cast<K*>(k);return Guard(p->api.a,[&]{p->Compute(ctx);});};
  op.KernelDestroy=[](void* k){delete static_cast<K*>(k);};
  if(conv){op.GetName=[](const OrtCustomOp*){return "CachedConv2D";};op.GetInputTypeCount=[](const OrtCustomOp*)->size_t{return 3;};op.GetOutputTypeCount=[](const OrtCustomOp*)->size_t{return 1;};op.GetOutputType=[](const OrtCustomOp*,size_t){return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;};}
  else {op.GetName=[](const OrtCustomOp*){return "MaxWithIndex";};op.GetInputTypeCount=[](const OrtCustomOp*)->size_t{return 1;};op.GetOutputTypeCount=[](const OrtCustomOp*)->size_t{return 2;};op.GetOutputType=[](const OrtCustomOp*,size_t i){return i==0?ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;};}
  return op;
}
OrtCustomOp conv_op=MakeOp<Conv>("CachedConv2D",3,1,true),max_op=MakeOp<Max>("MaxWithIndex",1,2,false);
std::mutex domain_mutex;OrtCustomOpDomain* domain=nullptr;
}
OPS_EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options,const OrtApiBase* base){
  const OrtApi* a=base->GetApi(16);if(!a)return base->GetApi(1)->CreateStatus(ORT_FAIL,"onnxruntime-ops requires ORT API 16 or newer");
  return Guard(a,[&]{std::lock_guard<std::mutex> lock(domain_mutex);if(!domain){OrtCustomOpDomain* p=nullptr;Api api{a};api.Check(a->CreateCustomOpDomain("onnxruntime_ops",&p));try{api.Check(a->CustomOpDomain_Add(p,&conv_op));api.Check(a->CustomOpDomain_Add(p,&max_op));}catch(...){a->ReleaseCustomOpDomain(p);throw;}domain=p;}Api{a}.Check(a->AddCustomOpDomain(options,domain));});
}
OPS_EXPORT const char* OrtOpsBuildInfo(){static const std::string info=std::string(R"({"version":"0.2.0","classic_conv":true,"ort_api":16,"cuda":)")+std::to_string(CUDART_VERSION)+R"(,"cudnn_major":)"+std::to_string(CUDNN_MAJOR)+"}";return info.c_str();}
OPS_EXPORT const char* OrtOpsStats(){static thread_local std::string value;value=std::string(R"({"builds":)")+std::to_string(builds.load())+R"(,"hits":)"+std::to_string(hits.load())+R"(,"evictions":)"+std::to_string(evictions.load())+",\"graph_ns\":"+std::to_string(graph_ns.load())+",\"heuristics_ns\":"+std::to_string(heuristics_ns.load())+",\"support_ns\":"+std::to_string(support_ns.load())+",\"plan_ns\":"+std::to_string(plan_ns.load())+",\"classic_builds\":"+std::to_string(classic_builds.load())+",\"classic_fallbacks\":"+std::to_string(classic_fallbacks.load())+"}";return value.c_str();}
