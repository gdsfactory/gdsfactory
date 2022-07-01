import asyncio
from pathlib import Path


async def handle_return(
    out_or_err: asyncio.streams.StreamReader,
    log_file: Path = None,
    to_console: bool = True,
) -> None:
    with open(log_file, "w") as f:
        while True:
            # Without this sleep, the program won't exit
            await asyncio.sleep(0)
            data = await out_or_err.readline()
            line = data.decode().strip()
            if line:
                if to_console:
                    print(line)
                f.write(line + "\n")


async def execute_and_stream_output(
    command: str, log_file_dir: Path, log_file_str: str
) -> asyncio.subprocess.Process:
    # Best not to use shell, but I can't get create_subprocess_exec to work here
    proc = await asyncio.create_subprocess_shell(
        command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    asyncio.create_task(
        handle_return(proc.stdout, log_file=log_file_dir / f"{log_file_str}_out.log")
    )
    asyncio.create_task(
        handle_return(proc.stderr, log_file=log_file_dir / f"{log_file_str}_err.log")
    )

    # This needs to also handle the "wait_to_finish" flag
    await proc.wait()
    return proc
