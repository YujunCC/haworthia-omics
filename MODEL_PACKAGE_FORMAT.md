# Haworthia OMICS Model Package

The application does not provide a pretrained model. This format lets users back up or move
models they trained or otherwise have the right to use.

## Contents

A generated ZIP contains one root directory with:

- `model_base.pth`: a compatible `TemperamentOmicsNet` state dictionary;
- `haworthia_omics.db`: taxonomy labels and numeric prototypes, with zero image rows;
- `MODEL_MANIFEST.json`: architecture, model hash, size, and catalog counts;
- `PACKAGE_MANIFEST.json`: format and data-boundary declarations;
- `MODEL_PACKAGE_NOTICE.txt`: user responsibility notice.

An optional `MODEL_LICENSE.txt` may describe terms chosen by the model owner. The software's
Apache-2.0 license does not automatically apply to a model package.

## Export

Open `数据库总览`, expand `导出当前模型包`, confirm the rights notice, and select
`生成模型包`. Download both the ZIP and its `.sha256` file. Export requires a loaded model
and at least one numeric prototype.

## Import

Open `数据库总览`, expand `导入模型包`, select the ZIP, paste the 64-character whole-package
SHA-256, confirm responsibility for the model and training data, and import.

The importer uses PyTorch's weights-only loader and validates the entire state dictionary.
It backs up the current database, model, and checkpoint. Matching taxa are merged by exact
`(species, variant)` identity; local images and extra local taxa remain in place. An existing
checkpoint is archived to prevent accidental resume against the newly imported model.

Technical compatibility and a matching hash do not prove that a model may lawfully be used
or redistributed. The exporting and importing users are responsible for model, image,
training-data, and license rights.
