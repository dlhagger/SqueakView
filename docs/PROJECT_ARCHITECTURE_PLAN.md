# SqueakView Project Architecture Plan

## Objective

Separate replaceable SqueakView application code from durable scientific project
data. A normal install, update, rollback, or clean application checkout must be
unable to modify or remove project settings, model packages, profiles,
qualification records, or run outputs.

This is a clean-break architecture for the next `main`. Existing repository-local
runtime data will not be migrated and legacy repository-relative data paths will
not be supported.

## Ownership model

### Application installation

The Git checkout at `~/Documents/SqueakView` is the application installation.
It contains only replaceable software, its local dependency environment, native
build products, and shipped resources:

```text
SqueakView/
├── squeakview/
├── native/
├── scripts/
├── resources/project_template/
├── build_engine/
├── tests/
├── pyproject.toml
└── uv.lock
```

Runtime code may read this tree. Normal application use must not write to it.
Dependency synchronization and native compilation are explicit installation or
development operations, not runtime operations.

### Scientific project

The default project parent is `~/Documents/SqueakView Projects`. Each project is
self-contained:

```text
MouseHouse/
├── squeakview_project.json
├── .squeakview.lock
├── runs/
├── models/
├── model_sources/
├── tasks/
├── profiles/
│   ├── experiments/
│   └── subjects/
└── qualification/
```

Project metadata and managed paths are schema-validated, size-bounded, and
resolved after symlinks. A managed path may not escape the project root.

### User and runtime state

User/device presentation state belongs under the platform configuration/state
directories rather than inside either the application or a project:

```text
~/.config/SqueakView/       recent project and GUI layout
~/.local/state/SqueakView/  launch logs produced before a project opens
$XDG_RUNTIME_DIR/squeakview/ supervisor sockets and hardware lock
```

Run-specific logs and diagnostics remain inside their immutable run directory.

## Non-negotiable invariants

1. Acquisition cannot start without an explicitly validated active project.
2. Normal runtime operation performs no writes beneath the application root.
3. Every run output is created beneath the active project's `runs/` directory.
4. Model packages, task overrides, profiles, and qualification state are owned by
   the active project.
5. Native plugins and shipped default templates are owned by the application.
6. Project files never depend on an absolute application-installation path.
7. Application updates never execute deletion, cleanup, or migration operations
   against a project.
8. One ownership-token lock prevents concurrent writers to a project.
9. One device-runtime lock prevents concurrent camera/controller acquisition,
   including from different projects.
10. Unknown or newer project schemas are rejected without modification.
11. Run manifests record project identity, application version/commit, selected
    configuration hashes, and device/runtime provenance.
12. Scientific capture, serial, validation, and shutdown behavior remain unchanged
    except for receiving explicit project paths.

## Target interfaces

### Path objects

Introduce immutable, Qt-free path objects:

- `AppPaths`: application root, native sources/builds, shipped resources, scripts.
- `ProjectPaths`: project metadata, runs, models, model sources, tasks, profiles,
  and qualification paths.
- `UserPaths`: configuration, persistent state/logs, and runtime paths.
- `RuntimeContext`: validated composition of the three, passed explicitly to
  services that read or write files.

The application root may still be derived from the installed module. Project and
user roots may not be derived from the application root or current directory.

### Project metadata

Initial schema:

```json
{
  "schema_version": 1,
  "project_id": "uuid",
  "name": "MouseHouse",
  "created_at": "RFC-3339 timestamp",
  "default_model": null,
  "default_task": "default.yaml"
}
```

Paths stored in project metadata are project-relative. Project creation is a
staged transaction: populate and validate a temporary sibling, then atomically
rename it into place.

### Startup workflow

1. Launch the project chooser without opening acquisition resources.
2. Resolve an explicit `--project`/`SQUEAKVIEW_PROJECT` selection, or offer the
   project launcher.
3. Create or open and validate the project.
4. Present Project Setup. Any TensorRT construction runs in a separate worker
   which owns the project lock and must exit before startup continues.
5. Select a published default model or explicitly continue without inference.
6. Pass the canonical project root to the durable supervisor, which acquires the
   complete-session project ownership lock.
7. Load project profiles, models, tasks, and run destination.
8. Acquire the device lock only when starting acquisition.

The chooser is shown on every startup with the most recently opened valid
project preselected. Project switching deliberately occurs only in this
pre-supervisor chooser: close SqueakView and relaunch to select another project.
There is no in-process project switch that could cross an acquisition or
finalization boundary.

## Implementation phases

### Current progress

- Phase 0: ownership model and path inventory documented.
- Phase 1: core path types, strict schema-1 metadata, staged non-overwriting
  creation, containment checks, ownership-token locks, and exclusive project
  sessions implemented.
- Phases 2–6: backend, GUI, launcher, preflight, model builder, qualification,
  and analysis entry points now receive explicit application/project/user paths;
  repo-local runtime fallbacks have been removed. Run-manifest schema 3 makes
  application and project identity mandatory. Automated ownership and isolation
  coverage is implemented.
- Phase 7: direct-checkout install/update workflow selected and documented;
  privileged device provisioning is separate from unprivileged dependency and
  native-build updates. First-time setup creates the default project parent only
  when absent and never changes an existing one. A two-application-root
  update/rollback test verifies that reopening a populated project leaves its
  complete file snapshot unchanged.
- Phase 8: the complete automated suite passes, including a representative
  create/profile/run workflow with the simulated application tree read-only.
  On 2026-09-16 a fresh project built and validated its project-owned YOLO26
  TensorRT package, then completed a triggered inference-enabled hardware smoke
  with 2,672 source, recording-admission, frame-manifest, and MP4 samples; zero
  frame gaps, incomplete frames, CRC failures, or recording evictions; validated
  controller/camera alignment; successful feeder-jam clear acknowledgement; and
  clean bounded finalization. A subsequent clean relaunch preselected the same
  project and restored its project-owned `mousehouse` default model, `DID`
  experiment, and `TEST` subject. A separately created `Isolation_Test` project
  received independent shipped sources/templates and a unique project ID while
  containing no models, profiles, or runs; the original project's model,
  profiles, and run remained intact. The remaining gates are explicitly divided
  below between repository automation and physical operator qualification.

### Phase 0 — Safety contract and inventory

- Add this architecture document and a machine-readable write-ownership test
  matrix.
- Inventory every existing repository-relative read and write.
- Classify each path as application, project, user-state, or runtime owned.
- Add regression tests around the current run/profile/model behavior before
  changing ownership.

Exit criterion: every mutable path has one declared future owner.

### Phase 1 — Project and path foundation

- Add strict `AppPaths`, `ProjectPaths`, `UserPaths`, and `RuntimeContext` types.
- Add schema-1 project metadata parsing, validation, and transactional creation.
- Add containment and symlink-escape checks.
- Add ownership-token project locking and device-runtime locking.
- Remove import-time directory creation from shared modules.

Exit criterion: temporary projects can be created, opened, locked, and rejected
safely in Qt-free tests.

### Phase 2 — Route backend persistence

- Inject project paths into run context, backend manager, manifests, storage
  checks, qualification services, and post-run tools.
- Move the acquisition lock from repository-local runs state to the runtime lock.
- Record project identity in each run manifest.
- Make subprocess commands receive canonical project/application roots explicitly.

Exit criterion: backend integration tests create all mutable outputs beneath a
temporary project while the simulated application root is read-only.

### Phase 3 — Route operator configuration

- Bind `ProfileStore`, model catalog, task selection, configuration dialogs, Open
  Run Folder, and disk meters to the active project.
- Store GUI layout and recent-project selection in user configuration.
- Remove workspace-anchor path remapping and legacy repository-local defaults.

Exit criterion: two test projects retain independent profiles, tasks, models, and
runs with no cross-project path leakage.

### Phase 4 — Project launcher and lifecycle

- Add Create Project and Open Project startup UI.
- Show the active project in the operator interface.
- Make project switching a close-and-relaunch operation through the mandatory
  pre-supervisor chooser; never switch an active backend's ownership roots.
- Provide clear handling for missing, locked, malformed, and unsupported projects.

Exit criterion: GUI tests cover create/open/reopen, launch-time switching,
locking, and error workflows.

### Phase 5 — Launcher, preflight, and model builder

- Update `squeakview.sh` to place launch logs in user state and pass an explicit
  project selection.
- Split one-time Jetson provisioning from unprivileged application installation.
- Make preflight distinguish application-native assets from project assets.
- Provide a pre-acquisition Project Setup screen which launches an isolated
  worker to consume project model sources and publish validated packages into
  that explicit project before the supervisor starts.
- Copy shipped defaults into new projects rather than editing them in place.

Exit criterion: a fresh project can rebuild a model and pass preflight without
writing to the application tree.

### Phase 6 — Remove legacy architecture

- Remove `SQUEAKVIEW_WORKSPACE`/`PRODUCT_WORKSPACE` as data-root concepts.
- Remove repository-local fallbacks for runs, models, profiles, tasks, and logs.
- Remove workspace-anchor remapping and all compatibility-only tests/docs.
- Ensure repository-local mutable directories are absent from a fresh checkout.

Exit criterion: starting without a project produces a clear launcher/error and
cannot silently recreate legacy directories.

### Phase 7 — Install/update workflow

- Keep the Git checkout at `~/Documents/SqueakView` as the application
  installation; do not copy releases into a second installation tree.
- Update in place with a fast-forward Git pull, `uv sync`, an unprivileged native
  rebuild, and the automated suite before scientific use.
- Roll back code by selecting a known-good commit or tag in the checkout, then
  repeat dependency synchronization, native compilation, and verification.
- Explicitly prohibit project paths as update/cleanup targets.

Exit criterion: updating or rolling back the checkout does not change a
project-tree hash, and normal runtime operation does not write to the checkout.

### Phase 8 — Qualification and handoff

- Run the entire automated suite.
- Test with the application tree read-only.
- Create a fresh MouseHouse project and rebuild its model package.
- Perform short start/stop, feeder-jam, GUI-loss, and validation tests.
- Perform an overnight scientific run only after the short matrix passes.
- Update the README and operator/install documentation for the new workflow.

Exit criterion: all output is project/user/runtime owned, no frames are lost, the
run finalizes correctly, and an application update/rollback leaves the project
unchanged.

### Phase 8 responsibility split

Codex/repository automation must complete without operator assistance:

- the full Python suite and diff/format checks;
- project creation, schema, containment, symlink-escape, and ownership tests;
- simulated read-only application-tree and two-application-root update/rollback
  tests;
- model-builder transaction, cancellation, worker-crash, and project-path tests;
- simulated GUI/supervisor IPC, lease-loss, startup cancellation, finalization,
  and failure-injection tests that do not signal live acquisition hardware;
- documentation and machine-readable ownership audit before commit.

The operator remains responsible only for evidence that inherently requires the
physical system:

- exercising a real camera/controller GUI-loss run on a safe bench;
- serial disconnect, hard-power-loss, and controller electrical-failsafe tests;
- one-hour and full-duration recordings with retained qualification evidence.

The post-reboot RP2040 USB enumeration issue is currently documented but
deferred. Kernel evidence shows that the controller is absent from the USB tree
until a physical reconnect; it is not treated as a completed SqueakView fix.

## User-input gates

Implementation should continue autonomously except at these product decisions:

1. Confirm the visible default project-parent name/location if it should differ
   from `~/Documents/SqueakView Projects`.
2. Resolved: always show the project launcher and preselect the most recently
   opened valid project.
3. Resolved: copy MouseHouse v2 and stock YOLO pose source packages into every
   new project's `model_sources/`; additional sources are project-owned.
4. Resolved: the Git checkout at `~/Documents/SqueakView` is the installation;
   projects default to the sibling `~/Documents/SqueakView Projects` tree and no
   second versioned-install tree is used.
5. Perform physical camera/controller tests and confirm observed GUI behavior at
   the Phase 8 hardware gates.

No existing runtime data will be copied, moved, renamed, or deleted as part of
this plan.
