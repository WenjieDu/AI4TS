"""JSON result and exit-code handling shared by data and analysis CLI commands."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import contextlib
import json
import sys

import click


def emit_json_result(action):
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = action()
        status = result["status"]
        code = (
            0
            if status in ("ready", "success")
            else 2 if status in ("needs_clarification", "needs_preparation", "partial") else 1
        )
    except Exception as error:
        result = dict(status="failed", error=f"{type(error).__name__}: {error}")
        code = 1
    click.echo(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if code:
        raise click.exceptions.Exit(code)


class JsonCommandGroup(click.Group):
    """Keep parsing failures in the command JSON contract."""

    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except click.ClickException as error:
            message = error.format_message()
            emit_json_result(lambda: dict(status="failed", error=message))
