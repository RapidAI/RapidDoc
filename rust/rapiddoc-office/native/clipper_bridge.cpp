// Thin adapter to the unmodified Clipper 6.4.2 core in clipper-sys 0.8.0.
// The crate's offset wrapper performs a preliminary boolean operation, which
// changes open strokes. Call the original API and preserve PolyTree DFS order.
#include "clipper.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>
struct RdPoint { std::int64_t x, y; };
struct RdPath { const RdPoint* points; std::size_t count; };
using RdVisit = int(*)(void*, int, const RdPoint*, std::size_t);
static ClipperLib::Path path(RdPath source) {
    ClipperLib::Path result; result.reserve(source.count);
    for (std::size_t i=0;i<source.count;++i) result.emplace_back(source.points[i].x,source.points[i].y);
    return result;
}
static int visit(const ClipperLib::PolyTree& tree, RdVisit callback, void* context) {
    std::vector<const ClipperLib::PolyNode*> stack;
    for(auto it=tree.Childs.rbegin();it!=tree.Childs.rend();++it) stack.push_back(*it);
    std::vector<RdPoint> points;
    while(!stack.empty()) {
        const auto* node=stack.back();stack.pop_back();points.clear();points.reserve(node->Contour.size());
        for(const auto& p:node->Contour) points.push_back({p.X,p.Y});
        if(!callback(context,node->IsHole(),points.data(),points.size())) return 2;
        for(auto it=node->Childs.rbegin();it!=node->Childs.rend();++it) stack.push_back(*it);
    }
    return 0;
}
extern "C" int rd_office_union(const RdPath* input,std::size_t count,int fill,RdVisit callback,void* context) noexcept {
    try {
        ClipperLib::Clipper clipper;bool any=false;
        for(std::size_t i=0;i<count;++i) any=clipper.AddPath(path(input[i]),ClipperLib::ptSubject,true)||any;
        if(!any) return 1;
        ClipperLib::PolyTree tree;
        auto rule=fill==0?ClipperLib::pftEvenOdd:ClipperLib::pftNonZero;
        if(!clipper.Execute(ClipperLib::ctUnion,tree,rule,rule)) return 1;
        return visit(tree,callback,context);
    }catch(...){return 1;}
}
extern "C" int rd_office_offset(RdPath input,int join,int end,double miter,double delta,RdVisit callback,void* context) noexcept {
    try {
        ClipperLib::ClipperOffset offset(miter,25.6);
        offset.AddPath(path(input),static_cast<ClipperLib::JoinType>(join),static_cast<ClipperLib::EndType>(end));
        ClipperLib::PolyTree tree;offset.Execute(tree,delta);
        return visit(tree,callback,context);
    }catch(...){return 1;}
}
