"""Pure metadata identity checks in isolated processes without numerical imports."""

# Created by Wenjie Du <wdu@time-series.ai>
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import subprocess
import sys
import unittest


PROJECT = Path(__file__).resolve().parents[1]
CHILD = r'''
import hashlib,json,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,sys.argv[1])
blocked={'pypots','numpy','pandas','scipy','sklearn','statsmodels','sktime','torch','transformers','pyarrow','safetensors','tensorflow','jax','keras'}
def audit(event,args):
    if event.startswith('socket.') or event in {'urllib.Request','http.client.connect'}:
        raise RuntimeError('Network forbidden')
    if event=='import' and args[0].split('.')[0] in blocked:
        raise RuntimeError('Metadata imported a backend: '+args[0])
    if event=='exec':
        filename=args[0].co_filename.replace(chr(92),'/')
        if any('/site-packages/'+name+'/' in filename for name in blocked):
            raise RuntimeError('Metadata executed a backend')
sys.addaudithook(audit)
from ai4ts.analysis import capabilities
mode=sys.argv[2]
if mode=='missing':
    with patch.object(capabilities,'find_spec',return_value=None):
        try:capabilities.implementation_identity()
        except ModuleNotFoundError:pass
        else:raise AssertionError('Missing AI4TS was silently accepted')
    print(json.dumps({'missing_refused':True}))
else:
    class Distribution:
        version='fixture-version'
        def read_text(self,name):
            assert name=='RECORD'
            return 'fixture RECORD\n'
    if mode=='records':
        with patch.object(capabilities.metadata,'distribution',return_value=Distribution()):
            result=capabilities.implementation_identity()
        expected=hashlib.sha256(b'fixture RECORD\n').hexdigest()
        assert all(x=={'version':'fixture-version','record_sha256':expected} for x in result['packages'].values())
    else:
        result=capabilities.implementation_identity()
        assert all(x is None for x in result['packages'].values())
    assert result['schema']=='analysis-implementation-v1'
    assert set(result)=={'schema','code','packages'}
    assert set(result['packages'])=={'numpy','pandas','scipy','statsmodels'}
    for name in ('analysis.py','ingestion.py','pipeline.py','profile.py','semantics.py','preprocessing.py','registry.py'):
        path=Path(sys.argv[1])/'ai4ts/data_protocols'/name
        assert result['code']['ai4ts.data_protocols.'+path.name]==hashlib.sha256(path.read_bytes()).hexdigest()
    assert not blocked.intersection(sys.modules)
    print(json.dumps({'schema':result['schema'],'backends_imported':False}))
'''


class TestMetadataIdentity(unittest.TestCase):
    def check_mode(self, mode):
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-S", "-c", CHILD, str(PROJECT), mode],
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_active_package_location_without_importing_it_or_backends(self):
        self.assertFalse(self.check_mode("identity")["backends_imported"])

    def test_optional_dependency_records_keep_schema_and_digests(self):
        self.assertEqual(self.check_mode("records")["schema"], "analysis-implementation-v1")

    def test_missing_ai4ts_fails_explicitly_without_import(self):
        self.assertTrue(self.check_mode("missing")["missing_refused"])


if __name__ == "__main__":
    unittest.main()
