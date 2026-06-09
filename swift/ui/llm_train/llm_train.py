import collections
import os
import re
import sys
import time
from functools import partial
from subprocess import PIPE, STDOUT, Popen
from typing import Dict, Type

import gradio as gr
import json
import torch
from gradio import Accordion, Checkbox, Dropdown, Slider, Tab, Textbox
from json import JSONDecodeError

from swift.llm import RLHFArguments
from swift.ui.base import BaseUI
from swift.ui.llm_train.advanced import Advanced
from swift.ui.llm_train.dataset import Dataset
from swift.ui.llm_train.galore import Galore
from swift.ui.llm_train.hyper import Hyper
from swift.ui.llm_train.lisa import Lisa
from swift.ui.llm_train.llamapro import LlamaPro
from swift.ui.llm_train.lora import LoRA
from swift.ui.llm_train.model import Model
from swift.ui.llm_train.quantization import Quantization
from swift.ui.llm_train.rlhf import RLHF
from swift.ui.llm_train.runtime import Runtime
from swift.ui.llm_train.save import Save
from swift.ui.llm_train.self_cog import SelfCog
from swift.utils import get_logger

logger = get_logger()

is_spaces = True if 'SPACE_ID' in os.environ else False
if is_spaces:
    is_shared_ui = True if 'modelscope/swift' in os.environ['SPACE_ID'] else False
else:
    is_shared_ui = False


class LLMTrain(BaseUI):
    group = 'llm_train'

    is_studio = os.environ.get('MODELSCOPE_ENVIRONMENT') == 'studio'

    sub_ui = [
        Model,
        Dataset,
        Runtime,
        Save,
        LoRA,
        Hyper,
        Quantization,
        SelfCog,
        Advanced,
        RLHF,
        Lisa,
        Galore,
        LlamaPro,
    ]

    locale_dict: Dict[str, Dict] = {
        'llm_train': {
            'label': {
                'zh': 'LLM训练',
                'en': 'LLM Training',
            }
        },
        'train_type': {
            'label': {
                'zh': '训练Stage',
                'en': 'Train Stage'
            },
            'info': {
                'zh': '请注意选择于此匹配的数据集，人类对齐配置在页面下方',
                'en': 'Please choose matched dataset, RLHF settings is at the bottom of the page'
            }
        },
        'submit_alert': {
            'value': {
                'zh':
                '任务已开始，请查看tensorboard或日志记录，关闭本页面不影响训练过程',
                'en':
                'Task started, please check the tensorboard or log file, '
                'closing this page does not affect training'
            }
        },
        'dataset_alert': {
            'value': {
                'zh': '请选择或填入一个数据集',
                'en': 'Please input or select a dataset'
            }
        },
        'submit': {
            'value': {
                'zh': '🚀 开始训练',
                'en': '🚀 Begin'
            }
        },
        'dry_run': {
            'label': {
                'zh': '仅生成运行命令',
                'en': 'Dry-run'
            },
            'info': {
                'zh': '仅生成运行命令，开发者自行运行',
                'en': 'Generate run command only, for manually running'
            }
        },
        'gpu_id': {
            'label': {
                'zh': '选择可用GPU',
                'en': 'Choose GPU'
            },
            'info': {
                'zh': '选择训练使用的GPU号，如CUDA不可用只能选择CPU',
                'en': 'Select GPU to train'
            }
        },
        'sft_type': {
            'label': {
                'zh': '训练方式',
                'en': 'Train type'
            },
            'info': {
                'zh': '选择训练的方式',
                'en': 'Select the training type'
            }
        },
        'seed': {
            'label': {
                'zh': '随机数种子',
                'en': 'Seed'
            },
            'info': {
                'zh': '选择随机数种子',
                'en': 'Select a random seed'
            }
        },
        'dtype': {
            'label': {
                'zh': '训练精度',
                'en': 'Training Precision'
            },
            'info': {
                'zh': '选择训练精度',
                'en': 'Select the training precision'
            }
        },
        'use_ddp': {
            'label': {
                'zh': '使用DDP',
                'en': 'Use DDP'
            },
            'info': {
                'zh': '是否使用数据并行训练',
                'en': 'Use Distributed Data Parallel to train'
            }
        },
        'ddp_num': {
            'label': {
                'zh': 'DDP分片数量',
                'en': 'Number of DDP sharding'
            },
            'info': {
                'zh': '启用多少进程的数据并行',
                'en': 'The data_1 parallel size of DDP'
            }
        },
        'tuner_backend': {
            'label': {
                'zh': 'Tuner backend',
                'en': 'Tuner backend'
            },
            'info': {
                'zh': 'tuner实现框架，建议peft或者unsloth',
                'en': 'The tuner backend, suggest to use peft or unsloth'
            }
        },
        'use_liger': {
            'label': {
                'zh': '使用Liger kernel',
                'en': 'Use Liger kernel'
            },
            'info': {
                'zh': 'Liger kernel可以有效降低显存使用',
                'en': 'Liger kernel can reduce memory usage'
            }
        },
        'sequence_parallel_size': {
            'label': {
                'zh': '序列并行分段',
                'en': 'Sequence parallel size'
            },
            'info': {
                'zh': 'DDP条件下的序列并行（减小显存），需要安装ms-swift[seq_parallel]',
                'en': 'Sequence parallel when ddp, need to install ms-swift[seq_parallel]'
            }
        },
        'train_param': {
            'label': {
                'zh': '训练参数设置',
                'en': 'Train settings'
            },
        },
    }

    choice_dict = BaseUI.get_choices_from_dataclass(RLHFArguments)
    default_dict = BaseUI.get_default_value_from_dataclass(RLHFArguments)
    arguments = BaseUI.get_argument_names(RLHFArguments)

    @classmethod
    def do_build_ui(cls, base_tab: Type['BaseUI']):
        with gr.TabItem(elem_id='llm_train', label=''):
            gpu_count = 0
            default_device = 'cpu'
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                default_device = '0'
            with gr.Blocks():
                Model.build_ui(base_tab)
                Dataset.build_ui(base_tab)
                with gr.Accordion(elem_id='train_param', open=True):
                    with gr.Row():
                        gr.Dropdown(
                            elem_id='train_type', choices=['pretrain/sft', 'rlhf'], value='pretrain/sft', scale=3)
                        gr.Dropdown(elem_id='sft_type', scale=2)
                        gr.Dropdown(elem_id='tuner_backend', scale=2)
                        gr.Textbox(elem_id='sequence_parallel_size', scale=3)
                    with gr.Row():
                        gr.Textbox(elem_id='seed', scale=4)
                        gr.Dropdown(elem_id='dtype', scale=4)
                        gr.Checkbox(elem_id='use_liger', scale=4)
                        gr.Checkbox(elem_id='use_ddp', value=False, scale=4)
                        gr.Textbox(elem_id='ddp_num', value='2', scale=4)
                Hyper.build_ui(base_tab)
                Runtime.build_ui(base_tab)
                with gr.Row():
                    gr.Dropdown(
                        elem_id='gpu_id',
                        multiselect=True,
                        choices=[str(i) for i in range(gpu_count)] + ['cpu'],
                        value=default_device,
                        scale=8)
                    if is_shared_ui:
                        gr.Checkbox(elem_id='dry_run', value=True, interactive=False, scale=4)
                    else:
                        gr.Checkbox(elem_id='dry_run', value=False, scale=4)
                    submit = gr.Button(elem_id='submit', scale=4, variant='primary')

                LoRA.build_ui(base_tab)
                RLHF.build_ui(base_tab)
                Quantization.build_ui(base_tab)
                Galore.build_ui(base_tab)
                Lisa.build_ui(base_tab)
                LlamaPro.build_ui(base_tab)
                SelfCog.build_ui(base_tab)
                Save.build_ui(base_tab)
                Advanced.build_ui(base_tab)

                cls.element('sft_type').change(
                    Hyper.update_lr, inputs=[base_tab.element('sft_type')], outputs=[cls.element('learning_rate')])

                cls.element('train_record').change(
                    partial(cls.update_all_settings, base_tab=base_tab),
                    inputs=[cls.element('model_type'), cls.element('train_record')],
                    outputs=[value for value in cls.elements().values() if not isinstance(value, (Tab, Accordion))])
                if cls.is_studio:
                    submit.click(
                        cls.update_runtime, [],
                        [cls.element('runtime_tab'), cls.element('log')]).then(
                            cls.train_studio,
                            [value for value in cls.elements().values() if not isinstance(value, (Tab, Accordion))],
                            [cls.element('log')] + Runtime.all_plots + [cls.element('running_cmd')],
                            queue=True)
                else:
                    submit.click(
                        cls.train_local,
                        [value for value in cls.elements().values() if not isinstance(value, (Tab, Accordion))], [
                            cls.element('running_cmd'),
                            cls.element('logging_dir'),
                            cls.element('runtime_tab'),
                            cls.element('running_tasks'),
                            cls.element('train_record'),
                        ],
                        queue=True)
                if not cls.is_studio:
                    base_tab.element('running_tasks').change(
                        partial(Runtime.task_changed, base_tab=base_tab), [base_tab.element('running_tasks')],
                        [value for value in base_tab.elements().values() if not isinstance(value, (Tab, Accordion))]
                        + [cls.element('log')] + Runtime.all_plots,
                        cancels=Runtime.log_event)
                    Runtime.element('kill_task').click(
                        Runtime.kill_task,
                        [Runtime.element('running_tasks')],
                        [Runtime.element('running_tasks')] + [Runtime.element('log')] + Runtime.all_plots,
                        cancels=[Runtime.log_event],
                    ).then(Runtime.reset, [], [Runtime.element('logging_dir')] + [Hyper.element('output_dir')])

    @classmethod
    def update_runtime(cls):
        return gr.update(open=True), gr.update(visible=True)

    @classmethod
    def update_all_settings(cls, model_type, train_record, base_tab):
        if not train_record:
            return [gr.update()] * len(base_tab.elements())
        cache = cls.load_cache(model_type, train_record)
        updates = []
        for key, value in base_tab.elements().items():
            if isinstance(value, (Tab, Accordion)):
                continue
            if (key in cache and isinstance(value, (Textbox, Dropdown, Slider, Checkbox)) and key != 'train_record'):
                updates.append(gr.update(value=cache[key]))
            else:
                updates.append(gr.update())
        return updates

    @classmethod
    def train(cls, *args):
        ignore_elements = ('model_type', 'logging_dir', 'more_params', 'train_type')
        default_args = cls.get_default_value_from_dataclass(RLHFArguments)
        kwargs = {}
        kwargs_is_list = {}
        other_kwargs = {}
        more_params = {}
        more_params_cmd = ''
        keys = [key for key, value in cls.elements().items() if not isinstance(value, (Tab, Accordion))]
        model_type = None
        do_rlhf = False
        for key, value in zip(keys, args):
            compare_value = default_args.get(key)
            if isinstance(value, str) and re.fullmatch(cls.int_regex, value):
                value = int(value)
            elif isinstance(value, str) and re.fullmatch(cls.float_regex, value):
                value = float(value)
            elif isinstance(value, str) and re.fullmatch(cls.bool_regex, value):
                value = True if value.lower() == 'true' else False
            if key not in ignore_elements and key in default_args and compare_value != value and value:
                kwargs[key] = value if not isinstance(value, list) else ' '.join(value)
                kwargs_is_list[key] = isinstance(value, list) or getattr(cls.element(key), 'is_list', False)
            else:
                other_kwargs[key] = value
            if key == 'more_params' and value:
                try:
                    more_params = json.loads(value)
                except (JSONDecodeError or TypeError):
                    more_params_cmd = value

            if key == 'model_type':
                model_type = value

            if key == 'train_type':
                do_rlhf = value == 'rlhf'

        if os.path.exists(kwargs['model_id_or_path']):
            kwargs['model_type'] = model_type

        kwargs.update(more_params)
        if 'dataset' not in kwargs and 'custom_train_dataset_path' not in kwargs:
            raise gr.Error(cls.locale('dataset_alert', cls.lang)['value'])

        cmd = 'rlhf' if do_rlhf else 'sft'
        if kwargs.get('deepspeed'):
            more_params_cmd += f' --deepspeed {kwargs.pop("deepspeed")} '
        sft_args = RLHFArguments(
            **{
                key: value.split(' ') if kwargs_is_list.get(key, False) and isinstance(value, str) else value
                for key, value in kwargs.items()
            })
        params = ''

        sep = f'{cls.quote} {cls.quote}'
        for e in kwargs:
            if isinstance(kwargs[e], list):
                params += f'--{e} {cls.quote}{sep.join(kwargs[e])}{cls.quote} '
            elif e in kwargs_is_list and kwargs_is_list[e]:
                all_args = [arg for arg in kwargs[e].split(' ') if arg.strip()]
                params += f'--{e} {cls.quote}{sep.join(all_args)}{cls.quote} '
            else:
                params += f'--{e} {cls.quote}{kwargs[e]}{cls.quote} '
        params += more_params_cmd + ' '
        params += f'--add_output_dir_suffix False --output_dir {sft_args.output_dir} ' \
                  f'--logging_dir {sft_args.logging_dir} --ignore_args_error True'
        ddp_param = ''
        devices = other_kwargs['gpu_id']
        devices = [d for d in devices if d]
        if other_kwargs['use_ddp']:
            assert int(other_kwargs['ddp_num']) > 0
            ddp_param = f'NPROC_PER_NODE={int(other_kwargs["ddp_num"])}'
        assert (len(devices) == 1 or 'cpu' not in devices)
        gpus = ','.join(devices)
        cuda_param = ''
        if gpus != 'cpu':
            cuda_param = f'CUDA_VISIBLE_DEVICES={gpus}'

        log_file = os.path.join(sft_args.logging_dir, 'run.log')
        if sys.platform == 'win32':
            if cuda_param:
                cuda_param = f'set {cuda_param} && '
            if ddp_param:
                ddp_param = f'set {ddp_param} && '
            run_command = f'{cuda_param}{ddp_param}start /b swift sft {params} > {log_file} 2>&1'
        elif cls.is_studio:
            run_command = f'{cuda_param} {ddp_param} swift {cmd} {params}'
        else:
            run_command = f'{cuda_param} {ddp_param} nohup swift {cmd} {params} > {log_file} 2>&1 &'
        logger.info(f'Run training: {run_command}')
        if model_type:
            record = {}
            for key, value in zip(keys, args):
                if key in default_args or key in ('more_params', 'train_type', 'use_ddp', 'ddp_num', 'gpu_id'):
                    record[key] = value or None
            cls.save_cache(model_type, record)
        return run_command, sft_args, other_kwargs

    @classmethod
    def train_studio(cls, *args):
        run_command, sft_args, other_kwargs = cls.train(*args)
        if not other_kwargs['dry_run']:
            lines = collections.deque(maxlen=int(os.environ.get('MAX_LOG_LINES', 50)))
            process = Popen(run_command, shell=True, stdout=PIPE, stderr=STDOUT)
            with process.stdout:
                for line in iter(process.stdout.readline, b''):
                    line = line.decode('utf-8')
                    lines.append(line)
                    yield ['\n'.join(lines)] + Runtime.plot(run_command) + [run_command]
        else:
            yield [
                'Current is dryrun mode so you can only view the training cmd, please duplicate this space to '
                'do training or use with inference.'
            ] + [None] * len(Runtime.sft_plot) + [run_command]

    @classmethod
    def train_local(cls, *args):
        run_command, sft_args, other_kwargs = cls.train(*args)
        if not other_kwargs['dry_run']:
            os.makedirs(sft_args.logging_dir, exist_ok=True)
            os.system(run_command)
            time.sleep(1)  # to make sure the log file has been created.
            gr.Info(cls.locale('submit_alert', cls.lang)['value'])
        return run_command, sft_args.logging_dir, gr.update(open=True), Runtime.refresh_tasks(
            sft_args.output_dir), gr.update(choices=cls.list_cache(sft_args.model_type))
