from contextlib import contextmanager
import joblib.parallel
from joblib.parallel import BatchCompletionCallBack  # 用于替换回调


@contextmanager
def tqdm_joblib(tqdm_object):
    # 定义新的回调类，更新进度条
    class TqdmBatchCompletionCallback(BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    # 替换原来的回调类
    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        # 恢复原始回调
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()
