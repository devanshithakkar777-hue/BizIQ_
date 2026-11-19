# Adding a module to BizIQ

1. Create `biziq/modules/yourmodule.py` (or add to an existing file) with a run function and a recommendation function.
2. Register it and import the file in `biziq/modules/__init__.py` (the import order is the run order).
3. Add sample data and a schema in `data_layer.py` if it needs a new dataset. Run `pytest`: the contract test picks it up automatically.

```python
from .. import registry
from ..core import ModuleResult

def run_example(data, ctx):                       # ctx.as_of, ctx.today, ctx.modules (earlier results), ctx.settings
    df = data["sales"]
    res = ModuleResult(name="ExampleIQ")
    res.facts = {"orders": int(len(df))}          # every number the app may quote must be here
    res.add_flag("amber", "Something to look at", "Plain-language detail with the numbers in it.")
    res.tables["detail"] = df.head(10)
    res.notes.append("Assumption: ...")           # say what you assumed
    return res

def rec_example(facts, flags):
    return [f"There were {facts['orders']} orders."]   # numbers must trace to facts or flag text (the number guard checks this)

registry.register(registry.ModuleSpec(
    name="ExampleIQ", group="Commercial", blurb="One line shown at the top of the page.",
    needs=["sales"], run=run_example, recommend=rec_example, audience="Both", sensitive=False))
```

## The contract (enforced by `registry.check_result`)
* `name` matches the spec; `facts` is not empty and has no NaN or infinite values.
* Every flag has a valid level (`red`, `amber`, `green`) and a title.
* Tables are DataFrames.
* Recommendation text may only contain numbers that appear in the facts or flag text (the number guard).

## Failure isolation
If a module raises or breaks the contract, it is recorded as failed. The other modules still run and the app stays up.

## Sensitive modules
Set `sensitive=True` for anything about individual people (salary, performance). Only roles with `view_people` can open it,
and it is removed from the executive summary and reports for everyone else.
