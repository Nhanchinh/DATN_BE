"""
Test script để verify evaluation system
Chạy file này để kiểm tra xem tất cả modules import được không
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_imports():
    """Test all evaluation system imports"""
    print("=" * 50)
    print("Testing Evaluation System Imports")
    print("=" * 50)
    
    try:
        print("\n✓ Testing models...")
        from app.models.evaluation import (
            EvaluationDocument,
            EvaluationMetrics,
            OverallMetrics
        )
        print("  SUCCESS: Models imported")
        
        print("\n✓ Testing schemas...")
        from app.schemas.evaluation import (
            EvaluateSingleRequest,
            EvaluateBatchRequest,
            CompareModelsRequest
        )
        print("  SUCCESS: Schemas imported")
        
        print("\n✓ Testing evaluation service...")
        from app.services.evaluation_service import EvaluationService
        print("  SUCCESS: Evaluation service imported")
        
        print("\n✓ Testing dataset service...")
        from app.services.dataset_service import DatasetService
        print("  SUCCESS: Dataset service imported")
        
        print("\n✓ Testing repository...")
        from app.repositories.evaluation_repository import EvaluationRepository
        print("  SUCCESS: Repository imported")
        
        print("\n✓ Testing router...")
        from app.routers.evaluation import router
        print("  SUCCESS: Router imported")
        
        print("\n" + "=" * 50)
        print("✅ ALL IMPORTS SUCCESSFUL!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ IMPORT FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_service():
    """Test evaluation service basic functionality"""
    print("\n" + "=" * 50)
    print("Testing Evaluation Service")
    print("=" * 50)
    
    try:
        from app.services.evaluation_service import EvaluationService
        
        service = EvaluationService()
        print("\n✓ EvaluationService initialized")
        
        # Test word segmentation
        texts = ["Sinh viên đại học đang học tập chăm chỉ"]
        segmented = service.preprocess_vietnamese(texts)
        print(f"\n✓ Word segmentation works:")
        print(f"  Original: {texts[0]}")
        print(f"  Segmented: {segmented[0]}")
        
        print("\n✅ EVALUATION SERVICE TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("EVALUATION SYSTEM VERIFICATION")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test service
        service_ok = test_evaluation_service()
        
        if service_ok:
            print("\n" + "=" * 50)
            print("🎉 ALL TESTS PASSED!")
            print("=" * 50)
            print("\nNext steps:")
            print("1. Start the server: uvicorn app.main:app --reload")
            print("2. Check API docs: http://localhost:8000/docs")
            print("3. Test evaluation endpoints")
        else:
            print("\n⚠️  Service tests failed")
            sys.exit(1)
    else:
        print("\n⚠️  Import tests failed - check dependencies")
        sys.exit(1)
