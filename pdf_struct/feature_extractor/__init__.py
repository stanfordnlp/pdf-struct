from pdf_struct.feature_extractor.hocr_balance_sheet_ja import HOCRFeatureExtractor
from pdf_struct.feature_extractor.pdf_contract import PDFContractEnFeatureExtractor, PDFContractEnFeatureExtractorWithLM
from pdf_struct.feature_extractor.pdf_contract_ja import PDFContractJaFeatureExtractor, PDFContractJaFeatureExtractorWithLM
from pdf_struct.feature_extractor.text_contract import TextContractFeatureExtractor, TextContractFeatureExtractorWithLM


feature_extractors = {
    'HOCRFeatureExtractor': HOCRFeatureExtractor,
    'PDFContractEnFeatureExtractor': PDFContractEnFeatureExtractor,
    'PDFContractEnFeatureExtractorWithLM': PDFContractEnFeatureExtractorWithLM,
    'PDFContractJaFeatureExtractor': PDFContractJaFeatureExtractor,
    'PDFContractJaFeatureExtractorWithLM': PDFContractJaFeatureExtractorWithLM,
    'TextContractFeatureExtractor': TextContractFeatureExtractor,
    'TextContractFeatureExtractorWithLM': TextContractFeatureExtractorWithLM
}
