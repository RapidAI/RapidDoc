from rapid_doc.model.layout.rapid_layout_self import ModelType, RapidLayout, RapidLayoutInput

if __name__ == '__main__':

    cfg = RapidLayoutInput(model_type=ModelType.PP_DOCLAYOUT_PLUS_L, conf_thresh=0.4)
    model = RapidLayout(cfg=cfg)

    all_results = model(img_contents=[r"reader_order_07.png",])
    print(all_results)
    all_results[0].vis(r"layout_vis.png")