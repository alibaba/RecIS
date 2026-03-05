from recis.hooks.hook import Hook
from recis.utils.mos import Mos


class MosReporterEvalHook(Hook):
    def __init__(self, mos_uri):
        self.mos = Mos(mos_uri, model_bank_path=True)

    def after_eval(self, *args, **kwargs):
        self.mos.report_mos_metrics(
            is_train=False,
        )
