Hook System
===========

Basic Hooks
-----------



RecIS provides a rich Hook system to extend the training process:

.. currentmodule:: recis.hooks.hook

Hook
~~~~

.. autoclass:: Hook
   :members: before_train, after_train, before_step, after_step

.. currentmodule:: recis.hooks.logger_hook

LoggerHook
~~~~~~~~~~

.. autofunction:: recis.framework.metrics.add_metric

.. autoclass:: LoggerHook

.. currentmodule:: recis.hooks.profiler_hook

ProfilerHook
~~~~~~~~~~~~

.. autoclass:: ProfilerHook
   :members: __init__

.. currentmodule:: recis.hooks.filter_hook

HashTableFilterHook
~~~~~~~~~~~~~~~~~~~

.. autoclass:: HashTableFilterHook
   :members: __init__

Custom Hooks
------------

.. code-block:: python

   from recis.hooks import Hook
   
   class CustomHook(Hook):
       def __init__(self, custom_param):
           self.custom_param = custom_param
       
       def before_train(self, trainer):
           print(f"Training started with {self.custom_param}")
       
       def after_step(self, trainer):
           if trainer.state.global_step % 1000 == 0:
               # Execute custom logic every 1000 steps
               self.custom_logic(trainer)
       
       def custom_logic(self, trainer):
           # Custom logic implementation
           pass
   
   # Use custom hook
   custom_hook = CustomHook("my_parameter")
   trainer.add_hook(custom_hook)