from src.process.PostProcessingProcess import PostProcessingProcess
from src.process.RuntimeContext import (
    build_instances,
    build_runtime_context,
    create_run_logger,
    finalize_postprocessing,
    validate_runtime_context,
)
from src.process.WorkerProcess import WorkerProcess


def main():
    context = build_runtime_context()
    log = create_run_logger(context.output)
    log.info(f"Iniciando job run_tag={context.run_tag}.")
    validate_runtime_context(context, log)

    instances = build_instances(
        context.instance_dir,
        context.output,
        context.files,
        context.threads_limit,
        context.time_limit,
    )
    WorkerProcess(context.workers, log, run_tag=context.run_tag).run_parallel(
        instancies=instances, solver=context.method
    )

    postprocessing = PostProcessingProcess(log=log, output=context.output)
    finalize_postprocessing(postprocessing, context.run_tag, context.build_target)


if __name__ == "__main__":
    main()
