// Development-only Windows sampler. It launches and samples only its own child.
// Uses the installed Windows DbgHelp; no change to parser/runtime configuration.
#define NOMINMAX
#include <windows.h>
#include <dbghelp.h>
#include <tlhelp32.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#pragma comment(lib, "dbghelp.lib")

std::wstring quote(const std::wstring& s) {
    std::wstring out=L"\"";size_t slashes=0;
    for(wchar_t c:s) {
        if(c==L'\\') {++slashes;continue;}
        if(c==L'\"') out.append(slashes*2+1,L'\\');
        else out.append(slashes,L'\\');
        slashes=0;out+=c;
    }
    out.append(slashes*2,L'\\');out+=L'\"';return out;
}
std::string utf8(const wchar_t* value) {
    int n=WideCharToMultiByte(CP_UTF8,0,value,-1,nullptr,0,nullptr,nullptr);
    std::string out(n,'\0');WideCharToMultiByte(CP_UTF8,0,value,-1,out.data(),n,nullptr,nullptr);
    out.pop_back();return out;
}
int wmain(int argc,wchar_t** argv) {
    if(argc<3) {std::cerr<<"Usage: native_stack_sample output.tsv program [args...]\n";return 2;}
    std::wstring command;
    for(int i=2;i<argc;++i) {if(i>2)command+=L' ';command+=quote(argv[i]);}
    STARTUPINFOW startup{};startup.cb=sizeof(startup);PROCESS_INFORMATION process{};
    if(!CreateProcessW(nullptr,command.data(),nullptr,nullptr,FALSE,CREATE_NO_WINDOW,nullptr,nullptr,&startup,&process)) {
        std::cerr<<"CreateProcess failed: "<<GetLastError()<<"\n";return 2;
    }
    // Windows venv python.exe is a redirector that waits on the real Python
    // child. Follow only descendants of the process that this sampler launched.
    HANDLE root_process=process.hProcess,root_thread=process.hThread;
    for(int attempt=0;attempt<10;++attempt) {
        Sleep(20);DWORD descendant=0;
        HANDLE snapshot=CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS,0);
        PROCESSENTRY32W entry{};entry.dwSize=sizeof(entry);
        if(snapshot!=INVALID_HANDLE_VALUE) {
            if(Process32FirstW(snapshot,&entry)) do {
                if(entry.th32ParentProcessID==process.dwProcessId &&
                   (_wcsicmp(entry.szExeFile,L"python.exe")==0 ||
                    _wcsicmp(entry.szExeFile,L"pythonw.exe")==0)) {
                    descendant=entry.th32ProcessID;break;
                }
            } while(Process32NextW(snapshot,&entry));
            CloseHandle(snapshot);
        }
        if(!descendant)continue;
        HANDLE child=OpenProcess(PROCESS_QUERY_INFORMATION|PROCESS_VM_READ|SYNCHRONIZE|PROCESS_TERMINATE,FALSE,descendant);
        if(!child)break;
        HANDLE thread=nullptr;ULARGE_INTEGER oldest{};oldest.QuadPart=~0ull;
        snapshot=CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD,0);
        THREADENTRY32 t{};t.dwSize=sizeof(t);
        if(snapshot!=INVALID_HANDLE_VALUE) {
            if(Thread32First(snapshot,&t)) do {
                if(t.th32OwnerProcessID!=descendant)continue;
                HANDLE h=OpenThread(THREAD_SUSPEND_RESUME|THREAD_GET_CONTEXT|THREAD_QUERY_INFORMATION,FALSE,t.th32ThreadID);
                FILETIME created,exited,kernel,user;
                if(h && GetThreadTimes(h,&created,&exited,&kernel,&user)) {
                    ULARGE_INTEGER time{};time.LowPart=created.dwLowDateTime;time.HighPart=created.dwHighDateTime;
                    if(time.QuadPart<oldest.QuadPart) {if(thread)CloseHandle(thread);thread=h;oldest=time;h=nullptr;}
                }
                if(h)CloseHandle(h);
            } while(Thread32Next(snapshot,&t));
            CloseHandle(snapshot);
        }
        if(!thread) {CloseHandle(child);break;}
        if(process.hProcess!=root_process) {CloseHandle(process.hThread);CloseHandle(process.hProcess);}
        process.hProcess=child;process.hThread=thread;process.dwProcessId=descendant;
        attempt=-1; // Also handle a nested redirector, bounded by real descendants.
    }
    // Load PDBs while the child is alive. Deferred loading after process exit
    // cannot read its PE debug directory and can silently report export names.
    SymSetOptions(SYMOPT_UNDNAME|SYMOPT_LOAD_LINES|SYMOPT_FAIL_CRITICAL_ERRORS|SYMOPT_NO_PROMPTS);
    if(!SymInitialize(process.hProcess,"target\\release;target\\release\\examples",FALSE)) {
        std::cerr<<"SymInitialize failed: "<<GetLastError()<<"\n";
        TerminateProcess(process.hProcess,2);CloseHandle(process.hThread);CloseHandle(process.hProcess);
        if(process.hProcess!=root_process) {CloseHandle(root_thread);CloseHandle(root_process);}return 2;
    }
    Sleep(100);
    SymRefreshModuleList(process.hProcess);
    std::map<std::vector<DWORD64>,size_t> counts;size_t failed=0,sampled=0;
    auto start=std::chrono::steady_clock::now();
    auto refresh=start;
    while(WaitForSingleObject(process.hProcess,5)==WAIT_TIMEOUT) {
        auto now=std::chrono::steady_clock::now();
        if(now-start>std::chrono::seconds(180)) {TerminateProcess(process.hProcess,124);break;}
        if(now-refresh>std::chrono::milliseconds(250)) {SymRefreshModuleList(process.hProcess);refresh=now;}
        if(SuspendThread(process.hThread)==DWORD(-1)) {++failed;continue;}
        CONTEXT context{};context.ContextFlags=CONTEXT_FULL;
        std::vector<DWORD64> frames;
        if(GetThreadContext(process.hThread,&context)) {
            STACKFRAME64 frame{};
            frame.AddrPC.Offset=context.Rip;frame.AddrPC.Mode=AddrModeFlat;
            frame.AddrFrame.Offset=context.Rbp;frame.AddrFrame.Mode=AddrModeFlat;
            frame.AddrStack.Offset=context.Rsp;frame.AddrStack.Mode=AddrModeFlat;
            frames.push_back(context.Rip);
            for(int i=0;i<96;++i) {
                if(!StackWalk64(IMAGE_FILE_MACHINE_AMD64,process.hProcess,process.hThread,&frame,&context,
                    nullptr,SymFunctionTableAccess64,SymGetModuleBase64,nullptr) || !frame.AddrPC.Offset) break;
                if(frames.back()!=frame.AddrPC.Offset)frames.push_back(frame.AddrPC.Offset);
            }
        } else ++failed;
        ResumeThread(process.hThread);
        if(!frames.empty()) {++counts[frames];++sampled;}
    }
    DWORD exit_code=0;GetExitCodeProcess(process.hProcess,&exit_code);
    std::map<DWORD64,std::string> names;
    std::ofstream output(argv[1]);
    for(auto& [frames,count]:counts) {
        output<<count;
        for(auto address:frames) {
            auto found=names.find(address);
            if(found==names.end()) {
                alignas(SYMBOL_INFO) unsigned char buffer[sizeof(SYMBOL_INFO)+MAX_SYM_NAME]{};
                auto info=reinterpret_cast<SYMBOL_INFO*>(buffer);info->SizeOfStruct=sizeof(SYMBOL_INFO);info->MaxNameLen=MAX_SYM_NAME;
                DWORD64 displacement=0;IMAGEHLP_MODULE64 module{};module.SizeOfStruct=sizeof(module);
                std::string name="unknown";
                if(SymGetModuleInfo64(process.hProcess,address,&module))name=module.ModuleName;
                if(SymFromAddr(process.hProcess,address,&displacement,info))name+="!"+std::string(info->Name);
                else {char hex[32];sprintf_s(hex,"!0x%llx",address);name+=hex;}
                std::replace(name.begin(),name.end(),'\t',' ');
                found=names.emplace(address,name).first;
            }
            output<<'\t'<<found->second;
        }
        output<<'\n';
    }
    std::cout<<"samples="<<sampled<<" failed="<<failed<<" stacks="<<counts.size()<<" child_exit="<<exit_code
             <<" sampled_pid="<<process.dwProcessId
             <<" seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"\n";
    SymCleanup(process.hProcess);CloseHandle(process.hThread);CloseHandle(process.hProcess);
    if(process.hProcess!=root_process) {WaitForSingleObject(root_process,5000);CloseHandle(root_thread);CloseHandle(root_process);}
    return exit_code==0?0:1;
}
