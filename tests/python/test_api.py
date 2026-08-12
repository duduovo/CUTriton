import json
import tempfile
import unittest
from pathlib import Path

import cutriton


class PythonApiTest(unittest.TestCase):
    def test_compile_json_and_run_reference_gelu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gelu_graph.json"
            path.write_text(
                json.dumps(
                    {
                        "inputs": ["x"],
                        "outputs": ["y"],
                        "nodes": [
                            {
                                "name": "gelu",
                                "op_type": "Gelu",
                                "inputs": ["x"],
                                "outputs": ["y"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            engine = cutriton.compile(path, target="cpu_reference")
            output = engine.create_context().run({"x": [-1.0, 0.0, 1.0]})
            self.assertIn("y", output)
            self.assertEqual(len(output["y"]), 3)


if __name__ == "__main__":
    unittest.main()
