"""vLLM general plugin: registers speculative method 'copyflash' (copy + DFlash router).

Registration only stores a qualified name, so loading this plugin never imports the speculator unless the
method is used; any failure is logged and swallowed so other vLLM runs in the same environment are unaffected.
"""


def register():
    try:
        from vllm.v1.worker.gpu.spec_decode.speculator_config import register_speculative_method
        register_speculative_method('copyflash', 'copyflash.config.CopyFlashSpeculatorConfig')
    except Exception as e:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning('copyflash plugin not registered: %r', e)
