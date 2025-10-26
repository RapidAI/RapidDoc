class BlockType:
    IMAGE = 'image'
    TABLE = 'table'
    IMAGE_BODY = 'image_body'
    TABLE_BODY = 'table_body'
    IMAGE_CAPTION = 'image_caption'
    TABLE_CAPTION = 'table_caption'
    IMAGE_FOOTNOTE = 'image_footnote'
    TABLE_FOOTNOTE = 'table_footnote'
    TEXT = 'text'
    TITLE = 'title'
    INTERLINE_EQUATION = 'interline_equation'
    LIST = 'list'
    INDEX = 'index'
    DISCARDED = 'discarded'

    # Added in vlm 2.5
    CODE = "code"
    CODE_BODY = "code_body"
    CODE_CAPTION = "code_caption"
    ALGORITHM = "algorithm"
    REF_TEXT = "ref_text"
    PHONETIC = "phonetic"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    ASIDE_TEXT = "aside_text"
    PAGE_FOOTNOTE = "page_footnote"


class ContentType:
    IMAGE = 'image'
    TABLE = 'table'
    TEXT = 'text'
    INTERLINE_EQUATION = 'interline_equation'
    INLINE_EQUATION = 'inline_equation'
    EQUATION = 'equation'
    CHECKBOX = 'checkbox'
    CODE = 'code'


class CategoryId:
    Title = 0
    Text = 1
    Abandon = 2
    ImageBody = 3
    ImageCaption = 4
    TableBody = 5
    TableCaption = 6
    TableFootnote = 7
    InterlineEquation_Layout = 8
    InterlineEquationNumber_Layout = 9
    InlineEquation = 13
    InterlineEquation_YOLO = 14
    OcrText = 15
    LowScoreText = 16
    ImageFootnote = 101
    CheckBox = 200


class MakeMode:
    MM_MD = 'mm_markdown'
    NLP_MD = 'nlp_markdown'
    CONTENT_LIST = 'content_list'


class ModelPath:
    pass

class SplitFlag:
    CROSS_PAGE = 'cross_page'
    LINES_DELETED = 'lines_deleted'

class ImageType:
    PIL = 'pil_img'
    BASE64 = 'base64_img'