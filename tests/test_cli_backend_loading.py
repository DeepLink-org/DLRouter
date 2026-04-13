"""Tests for definition-driven backend loading in the CLI."""

from types import SimpleNamespace

from dlrouter import __main__ as cli
from dlrouter.backends.base import CLIArg


class _FakeDefinition:
    def get_cli_args(self) -> list[CLIArg]:
        return [
            CLIArg(
                name='custom_port',
                type=int,
                default=12345,
                help='Custom backend port',
            )
        ]


def test_add_backend_args_uses_backend_definition(monkeypatch):
    monkeypatch.setattr(
        cli,
        'get_backend_definition',
        lambda _backend_type: _FakeDefinition(),
        raising=False,
    )

    parser = cli.build_base_parser()
    cli.add_backend_args(parser, 'vllm')

    assert '--custom_port' in parser.format_help()


def test_extract_backend_config_uses_backend_definition_arg_names(monkeypatch):
    monkeypatch.setattr(
        cli,
        'get_backend_definition',
        lambda _backend_type: _FakeDefinition(),
        raising=False,
    )

    args = SimpleNamespace(custom_port=45678)

    assert cli.extract_backend_config(args, 'vllm') == {'custom_port': 45678}
