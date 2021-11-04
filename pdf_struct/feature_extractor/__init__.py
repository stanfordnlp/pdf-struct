from pdf_struct.feature_extractor.hocr_balance_sheet_ja import HOCRFeatureExtractor
from pdf_struct.feature_extractor.pdf_contract import PDFContractEnFeatureExtractor
from pdf_struct.feature_extractor.pdf_contract_ja import PDFContractJaFeatureExtractor
from pdf_struct.feature_extractor.text_contract import TextContractFeatureExtractor


feature_extractors = {
    'HOCRFeatureExtractor': HOCRFeatureExtractor,
    'PDFContractEnFeatureExtractor': PDFContractEnFeatureExtractor,
    'PDFContractJaFeatureExtractor': PDFContractJaFeatureExtractor,
    'TextContractFeatureExtractor': TextContractFeatureExtractor
}
