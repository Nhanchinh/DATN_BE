"""
Evaluation Repository - MongoDB operations cho evaluation data
Quản lý evaluation logs, history, và model comparison
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.database.connection import get_database
from app.models.evaluation import (
    EvaluationDocument,
    OverallMetrics,
    SummaryDataItem,
)

logger = logging.getLogger(__name__)


class EvaluationRepository:
    """
    Repository cho evaluation operations trong MongoDB.
    
    Collections:
    - evaluation_logs: Lưu kết quả đánh giá
    - datasets: Metadata của datasets
    """
    
    def __init__(self, db: AsyncIOMotorDatabase):
        """
        Args:
            db: MongoDB database instance
        """
        self.db = db
        self.evaluations = db.evaluation_logs
        self.datasets = db.datasets
    
    async def save_evaluation_result(
        self,
        user_id: str,
        model_name: str,
        dataset_name: str,
        summary_data: List[SummaryDataItem],
        overall_metrics: OverallMetrics,
        status: str = "completed"
    ) -> str:
        """
        Lưu kết quả evaluation vào MongoDB.
        
        Args:
            user_id: ID của user
            model_name: Tên model được đánh giá
            dataset_name: Tên dataset
            summary_data: List các kết quả đánh giá chi tiết
            overall_metrics: Tổng hợp metrics
            status: Status của evaluation (completed/failed)
            
        Returns:
            evaluation_id (string)
        """
        document: EvaluationDocument = {
            "user_id": user_id,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "created_at": datetime.utcnow(),
            "summary_data": summary_data,
            "overall_metrics": overall_metrics,
            "status": status,
            "error_message": None
        }
        
        result = await self.evaluations.insert_one(document)
        evaluation_id = str(result.inserted_id)
        
        logger.info(f"Saved evaluation result: {evaluation_id}")
        
        return evaluation_id
    
    async def update_evaluation_status(
        self,
        evaluation_id: str,
        status: str,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Cập nhật status của evaluation (cho background tasks).
        
        Args:
            evaluation_id: ID của evaluation
            status: New status (running/completed/failed)
            error_message: Error message nếu failed
            
        Returns:
            True if updated successfully
        """
        update_doc = {
            "status": status
        }
        
        if error_message:
            update_doc["error_message"] = error_message
        
        result = await self.evaluations.update_one(
            {"_id": ObjectId(evaluation_id)},
            {"$set": update_doc}
        )
        
        return result.modified_count > 0
    
    async def get_evaluation_by_id(
        self,
        evaluation_id: str
    ) -> Optional[EvaluationDocument]:
        """
        Lấy evaluation theo ID.
        
        Args:
            evaluation_id: ID của evaluation
            
        Returns:
            EvaluationDocument hoặc None
        """
        document = await self.evaluations.find_one({"_id": ObjectId(evaluation_id)})
        
        if document:
            document['_id'] = str(document['_id'])
        
        return document
    
    async def get_evaluation_history(
        self,
        user_id: str,
        limit: int = 10,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None
    ) -> List[EvaluationDocument]:
        """
        Lấy lịch sử evaluation của user.
        
        Args:
            user_id: ID của user
            limit: Số lượng records tối đa
            model_name: Filter theo model (optional)
            dataset_name: Filter theo dataset (optional)
            
        Returns:
            List of EvaluationDocuments
        """
        query = {"user_id": user_id}
        
        if model_name:
            query["model_name"] = model_name
        
        if dataset_name:
            query["dataset_name"] = dataset_name
        
        cursor = self.evaluations.find(query).sort("created_at", -1).limit(limit)
        results = await cursor.to_list(length=limit)
        
        # Convert ObjectId to string
        for doc in results:
            doc['_id'] = str(doc['_id'])
        
        logger.info(f"Retrieved {len(results)} evaluation records for user {user_id}")
        
        return results
    
    async def compare_models(
        self,
        user_id: str,
        model_names: List[str],
        dataset_name: str,
        limit: int = 10
    ) -> Dict[str, Dict]:
        """
        So sánh performance của nhiều models.
        
        Sử dụng MongoDB aggregation để tính average metrics.
        
        Args:
            user_id: ID của user
            model_names: List các models cần so sánh
            dataset_name: Dataset name
            limit: Số lượng evaluations gần nhất cho mỗi model
            
        Returns:
            Dict mapping model_name -> average metrics
        """
        comparisons = {}
        
        for model_name in model_names:
            pipeline = [
                {
                    "$match": {
                        "user_id": user_id,
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                        "status": "completed"
                    }
                },
                {
                    "$sort": {"created_at": -1}
                },
                {
                    "$limit": limit
                },
                {
                    "$group": {
                        "_id": "$model_name",
                        "avg_rouge1": {"$avg": "$overall_metrics.avg_rouge1"},
                        "avg_rouge2": {"$avg": "$overall_metrics.avg_rouge2"},
                        "avg_rougeL": {"$avg": "$overall_metrics.avg_rougeL"},
                        "avg_bleu": {"$avg": "$overall_metrics.avg_bleu"},
                        "avg_bert_score": {"$avg": "$overall_metrics.avg_bert_score"},
                        "count": {"$sum": 1}
                    }
                }
            ]
            
            cursor = self.evaluations.aggregate(pipeline)
            results = await cursor.to_list(length=1)
            
            if results:
                comparisons[model_name] = results[0]
            else:
                # No data for this model
                comparisons[model_name] = {
                    "_id": model_name,
                    "avg_rouge1": 0.0,
                    "avg_rouge2": 0.0,
                    "avg_rougeL": 0.0,
                    "avg_bleu": 0.0,
                    "avg_bert_score": 0.0,
                    "count": 0
                }
        
        logger.info(f"Model comparison completed for {len(model_names)} models")
        
        return comparisons
    
    async def get_overall_statistics(
        self,
        user_id: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Dict:
        """
        Lấy tổng hợp statistics.
        
        Args:
            user_id: Filter theo user (optional)
            model_name: Filter theo model (optional)
            
        Returns:
            Dict chứa overall statistics
        """
        match_stage = {"status": "completed"}
        
        if user_id:
            match_stage["user_id"] = user_id
        
        if model_name:
            match_stage["model_name"] = model_name
        
        pipeline = [
            {"$match": match_stage},
            {
                "$group": {
                    "_id": None,
                    "total_evaluations": {"$sum": 1},
                    "total_samples": {"$sum": "$overall_metrics.total_samples"},
                    "avg_rouge1": {"$avg": "$overall_metrics.avg_rouge1"},
                    "avg_rouge2": {"$avg": "$overall_metrics.avg_rouge2"},
                    "avg_rougeL": {"$avg": "$overall_metrics.avg_rougeL"},
                    "avg_bleu": {"$avg": "$overall_metrics.avg_bleu"},
                    "avg_bert_score": {"$avg": "$overall_metrics.avg_bert_score"}
                }
            }
        ]
        
        cursor = self.evaluations.aggregate(pipeline)
        results = await cursor.to_list(length=1)
        
        if results:
            return results[0]
        else:
            return {
                "total_evaluations": 0,
                "total_samples": 0,
                "avg_rouge1": 0.0,
                "avg_rouge2": 0.0,
                "avg_rougeL": 0.0,
                "avg_bleu": 0.0,
                "avg_bert_score": 0.0
            }


async def get_evaluation_repository() -> EvaluationRepository:
    """
    Dependency injection cho FastAPI.
    
    Returns:
        EvaluationRepository instance
    """
    db = get_database()
    return EvaluationRepository(db)
