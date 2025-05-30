# Copyright (c) AIoWay Authors - All Rights Reserved


from aioway.execs import ExecTracer


def test_module_ok(module, td):
    out = module(td)
    assert out.keys() == {"a", "b"}


def test_module_frame_ok(module, frame):
    out = module(frame.td)
    assert out.keys() == {"a", "b"}


def test_tracer_create(tracer):
    assert tracer.attrs.keys() == {"a"}


def test_tracer_map(tracer):
    tracer = tracer.map("RENAME", a="c")
    assert tracer.attrs.keys() == {"c"}


def test_tracer_join(tracer, frame):
    other = ExecTracer.create("FRAME", frame)
    tracer = tracer.join(other, "NESTED_LOOP", on="a")
    assert tracer.attrs.keys() == {"a"}


def test_module_map(tracer, module, module_attrs):
    result = tracer.map("MODULE", module=module, output=module_attrs)
    assert result.attrs == module_attrs
