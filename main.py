import ray
from utils import (
    parse_args,
    setup,
    load_config,
    tqdm_remote_get,
    setup_train_vae,
    seed_all
)
from tqdm import tqdm
from environment.meshUtils import (
    create_collision_mesh,
    create_grasp_object_urdf
)
from multiprocessing import Pool
from train import cotrain, train_vae, pretrain, pretrain_gn
from generate_data import (
    create_grasp_objects,
    generate_pretrain_data,
    generate_pretrain_imprint_data,
    generate_imprints
)
from pathlib import Path
from torch.multiprocessing import spawn

if __name__ == "__main__":
    args = parse_args()
    seed_all(args.seed)
    ray.init()
    if args.mode == 'collision_mesh':
        async_create_collision_mesh = ray.remote(create_collision_mesh)
        object_paths = list(Path(args.objects).rglob('*normalized.obj'))
        tqdm_remote_get(task_handles=[async_create_collision_mesh.remote(
            object_path) for object_path in object_paths],
            desc='creating collision meshs')
        exit()
    elif args.mode == 'urdf':
        object_paths = [str(path)
                        for path in Path(args.objects).rglob('*.obj')]
        object_paths = list(filter(lambda path: not path.endswith(
            "collision.obj"), object_paths))
        with Pool(48) as p:
            rv = list(tqdm(p.imap(create_grasp_object_urdf, object_paths),
                           total=len(object_paths)))
        exit()

    config = load_config(args.config)
    if args.mode == 'cotrain':
        cotrain(*setup(args, config))
    elif args.mode == 'vae':
        train_vae(*setup_train_vae(args, config))
    elif args.mode == 'pretrain':
        pretrain(args, config)
    elif args.mode == 'pretrain_gn':
        pretrain_gn(args, config)
    elif args.mode == 'pretrain_dataset':
        generate_pretrain_data(*setup(args, config))
    elif args.mode == 'pretrain_imprint_dataset':
        generate_pretrain_imprint_data(*setup(args, config))
    elif args.mode == 'grasp_objects':
        create_grasp_objects(*setup(args, config))
    elif args.mode == 'imprint_baseline':
        generate_imprints(*setup(args, config))
