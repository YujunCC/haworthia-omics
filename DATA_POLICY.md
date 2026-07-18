# Data And Model Boundary

The public repository and core release contain application source code and documentation
only. They contain no pretrained model, prototype catalog, database, image, image path,
training checkpoint, or third-party segmentation weight.

User images, masks, databases, checkpoints, imported models, locally trained weights, and
exported model packages are local user assets. They are excluded by `.gitignore` and are not
licensed under the application's Apache-2.0 license.

The model export interface copies only the current model `state_dict`, taxonomy labels, and
numeric prototypes into a portable ZIP. It creates a new sanitized SQLite catalog with zero
image rows and paths. It never exports source images, masks, or optimizer checkpoints.

The model import interface checks the whole-package SHA-256, ZIP paths and size, allowed
file list, model architecture and tensors, SQLite integrity and schema, prototype dimensions,
and absence of image rows. Technical validation does not establish copyright, training-data,
or redistribution rights. Exporters and importers are responsible for those rights.

IS-Net, U2Net, and other segmentation weights are third-party assets downloaded separately
by the user from their upstream distribution. They are not part of this project and are not
covered by Apache-2.0. See `THIRD_PARTY_NOTICES.md`.

The application stores runtime data locally and binds its launcher to `127.0.0.1`. It does
not upload images or models to an external service.
