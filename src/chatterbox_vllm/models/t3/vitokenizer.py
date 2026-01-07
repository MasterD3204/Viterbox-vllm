import logging
import os
from typing import List, Optional, Union

import torch
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizer


# Special tokens
SOT = "[START]"
EOT = "[STOP]"
UNK = "[UNK]"
SPACE = "[SPACE]"
SPECIAL_TOKENS = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]

# Vietnamese language token
VI_LANG = "[vi]"

logger = logging.getLogger(__name__)

class ViTokenizer(PreTrainedTokenizer):
    """
    Vietnamese Tokenizer cho ChatterboxTTS - tương thích với vLLM.
    Dựa trên BPE tokenizer với vocab mở rộng cho tiếng Việt.
    """
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(
        self,
        vocab_file: str,
        unk_token: str = UNK,
        pad_token: str = "[PAD]",
        sep_token: str = "[SEP]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        **kwargs
    ):
        """
        Khởi tạo ViTokenizer.
        
        Args:
            vocab_file: Đường dẫn đến file tokenizer_vi_expanded.json
            unk_token: Token cho từ không xác định
            pad_token: Token padding
            sep_token: Token phân tách
            cls_token: Token classification
            mask_token: Token mask
        """
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file)
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            sep_token=sep_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs
        )
        self.check_vocabset_sot_eot()
        
        # Cache các special token IDs
        self._sot_id = self.tokenizer.token_to_id(SOT)
        self._eot_id = self.tokenizer.token_to_id(EOT)
        self._unk_id = self.tokenizer.token_to_id(UNK)
        self._space_id = self.tokenizer.token_to_id(SPACE)
        self._vi_lang_id = self.tokenizer.token_to_id(VI_LANG)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str = None, **kwargs):
        """
        Khởi tạo tokenizer từ pretrained model hoặc đường dẫn.
        
        Args:
            pretrained_model_name_or_path: Đường dẫn đến thư mục chứa tokenizer
            **kwargs: Các tham số bổ sung
        """
        if pretrained_model_name_or_path is None:
            # Load từ thư mục hiện tại của file này
            vocab_file = os.path.join(os.path.dirname(__file__), "tokenizer_vi_expanded.json")
        else:
            # Load từ đường dẫn được chỉ định
            if os.path.isdir(pretrained_model_name_or_path):
                vocab_file = os.path.join(pretrained_model_name_or_path, "tokenizer_vi_expanded.json")
            else:
                vocab_file = pretrained_model_name_or_path
        
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"Không tìm thấy file tokenizer: {vocab_file}")
            
        return cls(vocab_file=vocab_file, **kwargs)

    def check_vocabset_sot_eot(self):
        """Kiểm tra vocab có chứa các special tokens cần thiết."""
        voc = self.tokenizer.get_vocab()
        assert SOT in voc, f"Thiếu token {SOT} trong vocab"
        assert EOT in voc, f"Thiếu token {EOT} trong vocab"
        assert VI_LANG in voc, f"Thiếu token {VI_LANG} trong vocab"

    def get_vocab(self):
        """Trả về vocabulary dictionary."""
        return self.tokenizer.get_vocab()

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenize text thành danh sách tokens.
        
        Args:
            text: Văn bản cần tokenize
            
        Returns:
            Danh sách các tokens
        """
        # Thay thế khoảng trắng bằng token [SPACE]
        text = text.replace(' ', SPACE)
        return self.tokenizer.encode(text).tokens

    def _convert_token_to_id(self, token: str) -> int:
        """Chuyển đổi token thành ID."""
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            return self._unk_id
        return token_id

    def _convert_id_to_token(self, index: int) -> str:
        """Chuyển đổi ID thành token."""
        token = self.tokenizer.id_to_token(index)
        if token is None:
            return UNK
        return token

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Chuyển đổi danh sách tokens thành chuỗi văn bản.
        
        Args:
            tokens: Danh sách tokens
            
        Returns:
            Chuỗi văn bản đã được decode
        """
        text = "".join(tokens)
        text = text.replace(' ', '')
        text = text.replace(SPACE, ' ')
        text = text.replace(EOT, '')
        text = text.replace(SOT, '')
        text = text.replace(UNK, '')
        # Loại bỏ language tags
        text = text.replace(VI_LANG, '')
        return text.strip()

    def encode_vietnamese(
        self, 
        text: str, 
        add_language_tag: bool = True,
        add_special_tokens: bool = True
    ) -> List[int]:
        """
        Encode văn bản tiếng Việt với language tag.
        
        Args:
            text: Văn bản tiếng Việt cần encode
            add_language_tag: Thêm tag [vi] vào đầu
            add_special_tokens: Thêm [START] và [STOP]
            
        Returns:
            Danh sách token IDs
        """
        tokens = self._tokenize(text)
        ids = [self._convert_token_to_id(t) for t in tokens]
        
        if add_language_tag and self._vi_lang_id is not None:
            ids = [self._vi_lang_id] + ids
            
        if add_special_tokens:
            ids = [self._sot_id] + ids + [self._eot_id]
            
        return ids

    @property
    def vocab_size(self) -> int:
        """Kích thước vocabulary."""
        return self.tokenizer.get_vocab_size()

    @property
    def max_token_id(self) -> int:
        """ID token lớn nhất trong vocab."""
        return max(self.tokenizer.get_vocab().values())
    
    @property
    def sot_token_id(self) -> int:
        """ID của token [START]."""
        return self._sot_id
    
    @property
    def eot_token_id(self) -> int:
        """ID của token [STOP]."""
        return self._eot_id
    
    @property
    def vi_lang_token_id(self) -> int:
        """ID của token [vi]."""
        return self._vi_lang_id

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        """
        Lưu vocabulary vào thư mục.
        
        Args:
            save_directory: Thư mục lưu
            filename_prefix: Prefix cho tên file
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
            
        filename = "tokenizer_vi_expanded.json"
        if filename_prefix:
            filename = f"{filename_prefix}_{filename}"
            
        save_path = os.path.join(save_directory, filename)
        self.tokenizer.save(save_path)
        
        return (save_path,)

