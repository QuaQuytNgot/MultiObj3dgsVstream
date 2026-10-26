"""Production HTTP/ABR/StreamSTGS composition entry point."""
from .config import load_config


def main(argv=None):
    config=load_config(argv)
    from .runner import run
    return run(config)


if __name__=='__main__':
    raise SystemExit(main())
