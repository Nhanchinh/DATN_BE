"""
Dataset Service - Load và quản lý VietNews dataset
Hỗ trợ cả dataset từ HuggingFace và custom uploads
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset

logger = logging.getLogger(__name__)


class DatasetService:
    """
    Service quản lý datasets cho evaluation.
    
    **Features**:
    - Load VietNews từ HuggingFace (tự động cache)
    - Load custom dataset từ .jsonl/.csv
    - Get test splits với sample size linh hoạt
    """
    
    def __init__(self):
        """Khởi tạo dataset service"""
        self.cache_dir = Path("./data/datasets")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vietnews_dataset = None
        
        logger.info(f"DatasetService initialized with cache_dir: {self.cache_dir}")
    
    def load_vietnews(self, force_reload: bool = False) -> None:
        """
        Load VietNews dataset từ HuggingFace.
        
        Dataset: https://huggingface.co/datasets/tindang/vietnews
        Format: {"article": "...", "abstract": "..."}
        
        Args:
            force_reload: Force reload dataset ignoring cache
        """
        if self.vietnews_dataset is not None and not force_reload:
            logger.info("VietNews dataset already loaded (using cached version)")
            return
        
        try:
            logger.info("Loading VietNews dataset from HuggingFace...")
            
            # Load dataset với cache
            self.vietnews_dataset = load_dataset(
                "tindang/vietnews",
                cache_dir=str(self.cache_dir)
            )
            
            logger.info(f"VietNews dataset loaded successfully")
            logger.info(f"Train size: {len(self.vietnews_dataset['train'])}")
            logger.info(f"Test size: {len(self.vietnews_dataset['test'])}")
            
        except Exception as e:
            logger.error(f"Failed to load VietNews dataset: {e}")
            raise
    
    def get_test_split(
        self, 
        sample_size: int = 100,
        shuffle: bool = True,
        seed: int = 42
    ) -> Tuple[List[str], List[str]]:
        """
        Lấy test split từ VietNews dataset.
        
        Args:
            sample_size: Số lượng samples cần lấy (default 100)
            shuffle: Có shuffle dataset không (default True)
            seed: Random seed cho reproducibility
            
        Returns:
            Tuple of (articles, summaries)
        """
        # Ensure dataset is loaded
        if self.vietnews_dataset is None:
            self.load_vietnews()
        
        # Get test split
        test_data = self.vietnews_dataset['test']
        
        # Shuffle if needed
        if shuffle:
            test_data = test_data.shuffle(seed=seed)
        
        # Limit to sample_size
        test_data = test_data.select(range(min(sample_size, len(test_data))))
        
        # Extract articles and abstracts
        articles = test_data['article']
        summaries = test_data['abstract']
        
        logger.info(f"Retrieved {len(articles)} samples from VietNews test split")
        
        return articles, summaries
    
    def load_custom_dataset(
        self, 
        file_path: str,
        text_column: str = "text",
        summary_column: str = "summary"
    ) -> Tuple[List[str], List[str]]:
        """
        Load custom dataset từ file .jsonl hoặc .csv.
        
        Args:
            file_path: Path to dataset file
            text_column: Tên cột chứa văn bản gốc
            summary_column: Tên cột chứa tóm tắt
            
        Returns:
            Tuple of (texts, summaries)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        logger.info(f"Loading custom dataset from: {file_path}")
        
        try:
            # Detect file type and load
            if file_path.suffix == '.jsonl':
                dataset = load_dataset('json', data_files=str(file_path))
            elif file_path.suffix == '.csv':
                dataset = load_dataset('csv', data_files=str(file_path))
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # Extract data
            data = dataset['train']  # load_dataset always uses 'train' split
            texts = data[text_column]
            summaries = data[summary_column]
            
            logger.info(f"Loaded {len(texts)} samples from custom dataset")
            
            return texts, summaries
            
        except Exception as e:
            logger.error(f"Failed to load custom dataset: {e}")
            raise
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, any]:
        """
        Lấy thông tin về dataset.
        
        Args:
            dataset_name: Tên dataset (vietnews hoặc custom)
            
        Returns:
            Dict chứa metadata
        """
        if dataset_name == "vietnews":
            if self.vietnews_dataset is None:
                self.load_vietnews()
            
            return {
                "name": "VietNews",
                "source": "vietnews",
                "total_samples": len(self.vietnews_dataset['test']),
                "description": "Vietnamese news summarization dataset from HuggingFace"
            }
        else:
            # For custom datasets, metadata should be stored in DB
            return {
                "name": dataset_name,
                "source": "custom",
                "total_samples": 0,
                "description": "Custom dataset"
            }


# Singleton instance
_dataset_service: Optional[DatasetService] = None


def get_dataset_service() -> DatasetService:
    """
    Dependency injection cho FastAPI.
    Singleton pattern để cache dataset.
    """
    global _dataset_service
    if _dataset_service is None:
        _dataset_service = DatasetService()
    return _dataset_service
