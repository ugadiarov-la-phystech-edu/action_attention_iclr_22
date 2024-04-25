from action_attention import utils
from action_attention.stack import Stack, Seeds, SacredLog, Sieve
from action_attention.stacks.model.train_cswm import InitModel, InitTransitionsLoader, InitPathLoader, Train, Eval
from action_attention.stacks.model.slot_correlation import MeasureSlotCorrelation
from action_attention import paths
from action_attention.constants import Constants

ex = utils.setup_experiment("config/cswm.json")
ex.add_config(paths.CFG_MODEL_CSWM)


@ex.capture()
def get_model_config(model_config):

    # turn variable names into constants
    d = {}
    utils.process_config_dict(model_config, d)
    return d


@ex.config
def config():

    seed = None
    use_hard_attention = False
    use_soft_attention = False
    device = "cuda:0"
    learning_rate = 5e-4
    batch_size = 64
    epochs = 100
    model_save_path = None
    model_load_path = None
    dataset_path = "data/rs_lift/train"
    num_episodes = 5000
    num_episode_steps = 50
    eval_dataset_path = "data/rs_lift/val"
    eval_num_episodes = 1000
    eval_num_episode_steps = 50
    eval_clip_length = 20
    num_workers = 4
    need_train = True
    viz_names = None
    project = "Test project"
    group = None
    run_name = "run-0"


@ex.automain
def main(seed, use_hard_attention, use_soft_attention, device, learning_rate, batch_size, epochs, model_save_path,
         model_load_path, dataset_path, num_episodes, num_episode_steps, eval_dataset_path, eval_num_episodes,
         eval_num_episode_steps, eval_clip_length, num_workers, need_train, project, run_name, group, viz_names):

    model_config = get_model_config()
    logger = utils.Logger()
    stack = Stack(logger)

    stack.register(Seeds(
        use_torch=True,
        device=device,
        seed=seed
    ))
    stack.register(InitModel(
        model_config=model_config,
        learning_rate=learning_rate,
        device=device,
        load_path=model_load_path,
        use_hard_attention=use_hard_attention,
        use_soft_attention=use_soft_attention
    ))

    if need_train:
        # train model
        stack.register(InitTransitionsLoader(
            root_path=dataset_path,
            num_episodes=num_episodes,
            num_episode_steps=num_episode_steps,
            batch_size=batch_size,
            num_workers=num_workers,
        ))
        stack.register(Train(
            epochs=epochs,
            device=device,
            model_save_path=model_save_path,
            project=project,
            group=group,
            run_name=run_name
        ))
        stack.register(SacredLog(
            ex=ex,
            keys=[Constants.LOSSES],
            types=[SacredLog.TYPE_LIST]
        ))
    else:
        stack.register(Sieve(
            keys={Constants.MODEL}
        ))

        # evaluate model
        for i in [1, 5, 10]:

            stack.register(InitPathLoader(
                root_path=eval_dataset_path,
                path_length=i,
                num_episodes=eval_num_episodes,
                num_episode_steps=eval_num_episode_steps,
                clip_length=eval_clip_length,
                batch_size=batch_size,
                num_workers=num_workers
            ))
            stack.register(Eval(
                device=device,
                batch_size=batch_size,
                num_steps=i,
                dedup=False
            ))
            keys = [*[Constants.HITS.name + "_at_{:d}".format(k) for k in Eval.HITS_AT], Constants.MRR]
            stack.register(SacredLog(
                ex=ex,
                keys=keys,
                types=[SacredLog.TYPE_SCALAR for _ in range(len(keys))],
                prefix="{:d}_step".format(i)
            ))
            stack.register(Sieve(
                keys={Constants.MODEL}
            ))

        # # calculate correlation between slots
        # stack.register(InitPathLoader(
        #     root_path=eval_dataset_path,
        #     path_length=10,
        #     batch_size=100,
        #     factored_actions=False
        # ))
        # stack.register(MeasureSlotCorrelation(
        #     device=device
        # ))
        # stack.register(SacredLog(
        #     ex=ex,
        #     keys=[Constants.CORRELATION],
        #     types=[SacredLog.TYPE_SCALAR]
        # ))

    stack.forward(None, viz_names)
