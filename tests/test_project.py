from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from squeakview.project import (
    AppPaths,
    OwnershipLock,
    ProjectMetadata,
    ProjectCatalog,
    ProjectPaths,
    ProjectSession,
    RuntimeContext,
    UserPaths,
    create_project,
    open_project,
    set_default_model,
    validate_external_output_path,
)
from squeakview.common import run_context
from squeakview.common.profiles import ExperimentProfile, ProfileStore, SubjectProfile


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _tree_snapshot(root: Path) -> tuple[tuple[str, str, int], ...]:
    return tuple(
        (
            path.relative_to(root).as_posix(),
            _sha256(path),
            path.stat().st_mode & 0o777,
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )


class ProjectArchitectureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.app_root = self.root / "app"
        template_tasks = self.app_root / "resources" / "project_template" / "tasks"
        template_tasks.mkdir(parents=True)
        (template_tasks / "default.yaml").write_text(
            "task_name: Test\nschema_version: 1\n",
            encoding="utf-8",
        )
        self.app = AppPaths.from_root(self.app_root)
        self.projects = self.root / "projects"
        self.projects.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_project_creation_publishes_complete_strict_layout(self) -> None:
        project = create_project(
            self.projects / "MouseHouse",
            name="MouseHouse",
            app=self.app,
        )

        self.assertEqual(project.metadata.schema_version, 1)
        self.assertEqual(project.metadata.name, "MouseHouse")
        self.assertTrue(project.paths.metadata.is_file())
        self.assertTrue(project.paths.ownership_lock.is_file())
        self.assertEqual(project.paths.ownership_lock.read_bytes(), b"")
        self.assertEqual(project.paths.ownership_lock.stat().st_mode & 0o777, 0o600)
        self.assertEqual(
            (project.paths.tasks / "default.yaml").read_text(encoding="utf-8"),
            "task_name: Test\nschema_version: 1\n",
        )
        for directory in (
            project.paths.runs,
            project.paths.models,
            project.paths.model_sources,
            project.paths.profiles,
            project.paths.profiles / "experiments",
            project.paths.profiles / "subjects",
            project.paths.qualification,
        ):
            self.assertTrue(directory.is_dir(), directory)
            self.assertEqual(directory.stat().st_mode & 0o777, 0o700)
        for file_path in project.paths.root.rglob("*"):
            if file_path.is_file():
                self.assertEqual(file_path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(open_project(project.paths.root), project)
        self.assertEqual(list(self.projects.glob(".*.creating-*")), [])

    def test_project_creation_copies_seed_assets_without_linking_to_app(self) -> None:
        source = self.app.project_template / "model_sources/mousehouse_v2/model.pt"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"seed-model")

        project = create_project(
            self.projects / "Seeded",
            name="Seeded",
            app=self.app,
        )
        copied = project.paths.model_sources / "mousehouse_v2/model.pt"

        self.assertEqual(copied.read_bytes(), b"seed-model")
        copied.write_bytes(b"project-owned")
        self.assertEqual(source.read_bytes(), b"seed-model")

    def test_project_workflow_succeeds_with_read_only_application_tree(self) -> None:
        paths = sorted(self.app_root.rglob("*"), key=lambda path: len(path.parts), reverse=True)
        for path in paths:
            path.chmod(0o444 if path.is_file() else 0o555)
        self.app_root.chmod(0o555)
        try:
            project = create_project(
                self.projects / "ReadOnlyApp",
                name="Read Only App",
                app=self.app,
            )
            self.assertTrue((project.paths.tasks / "default.yaml").is_file())
            ProfileStore(
                project.paths.profiles,
                project_paths=project.paths,
            ).save_subject(SubjectProfile(name="Mouse", subject_id="mouse"))
            run_dir, _ = run_context.create_run_dir(
                runs_dir=project.paths.runs,
                experiment_name="Study",
                mouse_id="Mouse",
            )
            self.assertTrue(run_dir.is_dir())
            self.assertEqual(
                ProfileStore(
                    project.paths.profiles,
                    project_paths=project.paths,
                ).list_subjects()[0].subject_id,
                "mouse",
            )
        finally:
            self.app_root.chmod(0o755)
            for path in sorted(self.app_root.rglob("*"), key=lambda path: len(path.parts)):
                path.chmod(0o644 if path.is_file() else 0o755)

    def test_real_template_ships_exactly_the_two_supported_model_sources(self) -> None:
        template = AppPaths.discover().project_template / "model_sources"
        expected = {
            "mousehouse_v2/mousehouse_v2.pt",
            "mousehouse_v2/mousehouse_v2.yaml",
            "stock_yolo26_pose/coco-pose.yaml",
            "stock_yolo26_pose/yolo26n-pose.pt",
        }
        actual = {
            path.relative_to(template).as_posix()
            for path in template.rglob("*")
            if path.is_file()
        }

        self.assertEqual(actual, expected)
        self.assertTrue(all((template / relative).stat().st_size > 0 for relative in expected))

    def test_real_template_creates_byte_identical_independent_seed_copies(self) -> None:
        app = AppPaths.discover()
        source_root = app.project_template / "model_sources"
        relative_files = (
            Path("mousehouse_v2/mousehouse_v2.pt"),
            Path("mousehouse_v2/mousehouse_v2.yaml"),
            Path("stock_yolo26_pose/yolo26n-pose.pt"),
            Path("stock_yolo26_pose/coco-pose.yaml"),
        )
        source_hashes = {
            relative: _sha256(source_root / relative)
            for relative in relative_files
        }

        project = create_project(
            self.projects / "ActualTemplate",
            name="Actual Template",
            app=app,
        )

        for relative, expected_hash in source_hashes.items():
            copied = project.paths.model_sources / relative
            self.assertEqual(_sha256(copied), expected_hash)
            self.assertNotEqual(copied.stat().st_ino, (source_root / relative).stat().st_ino)
            self.assertEqual(_sha256(source_root / relative), expected_hash)

    def test_machine_readable_write_ownership_contract_matches_project_paths(self) -> None:
        contract_path = AppPaths.discover().resources / "write_ownership.v1.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))

        self.assertEqual(contract["schema_version"], 1)
        self.assertEqual(contract["forbidden_runtime_write_owner"], "application")
        self.assertEqual(
            contract["owners"]["project"]["metadata_files"],
            ["squeakview_project.json"],
        )
        self.assertEqual(
            contract["owners"]["project"]["session_files"],
            [".squeakview.lock"],
        )
        self.assertEqual(
            set(contract["owners"]["project"]["relative_roots"]),
            {
                "runs/",
                "models/",
                "model_sources/",
                "tasks/",
                "profiles/",
                "qualification/",
            },
        )

    def test_runtime_output_path_rejects_application_and_project_trees(self) -> None:
        project = create_project(
            self.projects / "OutputBoundary",
            name="Output Boundary",
            app=self.app,
        )
        external = self.root / "user-state/operator.log"

        self.assertEqual(
            validate_external_output_path(
                external,
                app=self.app,
                project=project.paths,
                label="test log",
            ),
            external.resolve(),
        )
        with self.assertRaisesRegex(ValueError, "application checkout"):
            validate_external_output_path(
                self.app.root / "runtime.log",
                app=self.app,
                project=project.paths,
            )
        with self.assertRaisesRegex(ValueError, "scientific project"):
            validate_external_output_path(
                project.paths.root / "runtime.log",
                app=self.app,
                project=project.paths,
            )

    def test_creation_refuses_to_replace_any_existing_destination(self) -> None:
        destination = self.projects / "important"
        destination.mkdir()
        marker = destination / "keep.txt"
        marker.write_text("scientific data", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            create_project(destination, name="Replacement", app=self.app)

        self.assertEqual(marker.read_text(encoding="utf-8"), "scientific data")

    def test_creation_failure_does_not_publish_partial_project(self) -> None:
        empty_app_root = self.root / "empty-app"
        empty_app_root.mkdir()

        with self.assertRaisesRegex(ValueError, "template is missing"):
            create_project(
                self.projects / "NeverPublished",
                name="Broken",
                app=AppPaths.from_root(empty_app_root),
            )

        self.assertFalse((self.projects / "NeverPublished").exists())
        self.assertEqual(list(self.projects.glob(".*.creating-*")), [])

    def test_unknown_schema_is_rejected_without_modifying_metadata(self) -> None:
        project = create_project(
            self.projects / "Future",
            name="Future",
            app=self.app,
        )
        payload = json.loads(project.paths.metadata.read_text(encoding="utf-8"))
        payload["schema_version"] = 99
        original = json.dumps(payload, sort_keys=True)
        project.paths.metadata.write_text(original, encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "unsupported project schema 99"):
            open_project(project.paths.root)

        self.assertEqual(project.paths.metadata.read_text(encoding="utf-8"), original)

    def test_metadata_rejects_unknown_fields_and_noncanonical_ids(self) -> None:
        metadata = ProjectMetadata.create("MouseHouse")
        payload = {
            "schema_version": metadata.schema_version,
            "project_id": metadata.project_id.upper(),
            "name": metadata.name,
            "created_at": metadata.created_at,
            "default_model": None,
            "default_task": "default.yaml",
            "unexpected": True,
        }

        with self.assertRaisesRegex(ValueError, "unknown fields"):
            ProjectMetadata.from_mapping(payload)
        payload.pop("unexpected")
        with self.assertRaisesRegex(ValueError, "canonical UUID"):
            ProjectMetadata.from_mapping(payload)

    def test_default_model_update_is_atomic_private_and_reopenable(self) -> None:
        project = create_project(
            self.projects / "Defaults",
            name="Defaults",
            app=self.app,
        )
        package = project.paths.models / "mousehouse_v2"
        (package / "configs").mkdir(parents=True)
        (package / "configs/mousehouse_v2.txt").write_text(
            "[property]\n", encoding="utf-8"
        )
        (package / "model.yaml").write_text(
            "schema_version: 3\n", encoding="utf-8"
        )

        updated = set_default_model(project, "mousehouse_v2")

        self.assertEqual(updated.metadata.default_model, "mousehouse_v2")
        self.assertEqual(open_project(project.paths.root).metadata.default_model, "mousehouse_v2")
        self.assertEqual(project.paths.metadata.stat().st_mode & 0o777, 0o600)
        self.assertEqual(list(project.paths.root.glob(".squeakview_project.json.*.tmp")), [])

        with self.assertRaisesRegex(ValueError, "cannot be resolved"):
            set_default_model(updated, "missing")

    def test_managed_paths_reject_absolute_traversal_and_symlink_escape(self) -> None:
        project = create_project(
            self.projects / "Contained",
            name="Contained",
            app=self.app,
        )
        outside = self.root / "outside"
        outside.mkdir()
        (project.paths.models / "escaped").symlink_to(outside, target_is_directory=True)

        with self.assertRaisesRegex(ValueError, "must be relative"):
            project.paths.managed_path(outside)
        with self.assertRaisesRegex(ValueError, "traversal"):
            project.paths.managed_path("../outside")
        with self.assertRaisesRegex(ValueError, "escapes"):
            project.paths.managed_path("models/escaped/file.engine")

    def test_project_asset_paths_are_portable_and_category_bounded(self) -> None:
        project = create_project(
            self.projects / "Portable",
            name="Portable",
            app=self.app,
        )
        task = project.paths.tasks / "default.yaml"

        self.assertEqual(
            project.paths.resolve_path(
                "tasks/default.yaml",
                within=project.paths.tasks,
                must_exist=True,
            ),
            task,
        )
        self.assertEqual(project.paths.portable_path(task), "tasks/default.yaml")
        with self.assertRaisesRegex(ValueError, "required directory"):
            project.paths.resolve_path(task, within=project.paths.models)

    def test_runtime_context_rejects_overlapping_app_and_project_roots(self) -> None:
        nested = self.app_root / "project"
        user = UserPaths(
            config=self.root / "config",
            state=self.root / "state",
            runtime=self.root / "runtime",
            projects_parent=self.projects,
        )

        with self.assertRaisesRegex(ValueError, "non-overlapping"):
            RuntimeContext(
                app=self.app,
                project=create_project(
                    nested,
                    name="Nested",
                    app=self.app,
                ),
                user=user,
            )

    def test_user_state_cannot_overlap_application_or_project(self) -> None:
        project = create_project(
            self.projects / "Separated",
            name="Separated",
            app=self.app,
        )
        overlapping_app = UserPaths(
            config=self.app_root / "config",
            state=self.root / "state",
            runtime=self.root / "runtime",
            projects_parent=self.projects,
        )
        with self.assertRaisesRegex(ValueError, "application root"):
            RuntimeContext(app=self.app, project=project, user=overlapping_app)

        overlapping_project = UserPaths(
            config=project.paths.root / "user-config",
            state=self.root / "state",
            runtime=self.root / "runtime",
            projects_parent=self.projects,
        )
        with self.assertRaisesRegex(ValueError, "project root"):
            RuntimeContext(app=self.app, project=project, user=overlapping_project)

    def test_catalog_rejects_project_parent_that_contains_application(self) -> None:
        user = UserPaths(
            config=self.root / "config",
            state=self.root / "state",
            runtime=self.root / "runtime",
            projects_parent=self.root,
        )

        with self.assertRaisesRegex(ValueError, "project parent"):
            ProjectCatalog(app=self.app, user=user)

    def test_user_paths_are_independent_of_application_root(self) -> None:
        paths = UserPaths.discover(
            {
                "HOME": str(self.root / "home"),
                "XDG_CONFIG_HOME": str(self.root / "config-home"),
                "XDG_STATE_HOME": str(self.root / "state-home"),
                "XDG_RUNTIME_DIR": str(self.root / "runtime-home"),
            }
        )

        self.assertEqual(paths.config, (self.root / "config-home/SqueakView").resolve())
        self.assertEqual(paths.state, (self.root / "state-home/SqueakView").resolve())
        self.assertEqual(paths.runtime, (self.root / "runtime-home/squeakview").resolve())
        self.assertEqual(
            paths.projects_parent,
            (self.root / "home/Documents/SqueakView Projects").resolve(),
        )

    def test_project_session_excludes_second_writer_and_releases_cleanly(self) -> None:
        project = create_project(
            self.projects / "Exclusive",
            name="Exclusive",
            app=self.app,
        )
        first = ProjectSession.open(project.paths.root)
        self.addCleanup(first.close)

        with self.assertRaisesRegex(RuntimeError, "another SqueakView process"):
            ProjectSession.open(project.paths.root)

        first.close()
        second = ProjectSession.open(project.paths.root)
        self.assertTrue(second.active)
        second.close()
        self.assertFalse(second.active)


class OwnershipLockTests(unittest.TestCase):
    def test_lock_excludes_second_owner_and_records_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "project.lock"
            first = OwnershipLock(path, purpose="project")
            second = OwnershipLock(path, purpose="project")

            self.assertTrue(first.acquire())
            self.assertFalse(second.acquire())
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["purpose"], "project")
            self.assertEqual(payload["token"], first.token)
            self.assertGreater(payload["pid"], 0)
            first.release()
            self.assertTrue(second.acquire())
            second.release()

    def test_lock_refuses_symlink_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "target"
            target.write_text("do not overwrite", encoding="utf-8")
            link = root / "lock"
            link.symlink_to(target)

            with self.assertRaises(OSError):
                OwnershipLock(link, purpose="project").acquire()

            self.assertEqual(target.read_text(encoding="utf-8"), "do not overwrite")


class ProjectCatalogTests(unittest.TestCase):
    def test_catalog_creates_missing_default_parent_once_without_rewriting_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app_root = root / "app"
            app_root.mkdir()
            projects_parent = root / "nested/projects"
            user = UserPaths(
                config=root / "user/config",
                state=root / "user/state",
                runtime=root / "user/runtime",
                projects_parent=projects_parent,
            )
            catalog = ProjectCatalog(app=AppPaths.from_root(app_root), user=user)

            created = catalog.ensure_projects_parent()
            self.assertEqual(created, projects_parent.resolve())
            self.assertTrue(created.is_dir())
            self.assertEqual(created.stat().st_mode & 0o777, 0o700)

            marker = created / "existing-project-data"
            marker.write_text("preserve", encoding="utf-8")
            created.chmod(0o750)
            before = created.stat()

            reopened = catalog.ensure_projects_parent()

            after = reopened.stat()
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")
            self.assertEqual(after.st_mode & 0o777, 0o750)
            self.assertEqual(after.st_mtime_ns, before.st_mtime_ns)

    def test_catalog_creates_and_remembers_projects_outside_the_app(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app_root = root / "app"
            (app_root / "resources/project_template/tasks").mkdir(parents=True)
            (app_root / "resources/project_template/tasks/default.yaml").write_text(
                "task_name: Test\n"
            )
            user = UserPaths(
                config=root / "user/config",
                state=root / "user/state",
                runtime=root / "user/runtime",
                projects_parent=root / "projects",
            )
            catalog = ProjectCatalog(app=AppPaths.from_root(app_root), user=user)

            project = catalog.create(name="Mouse House")

            self.assertEqual(project.paths.root, root / "projects/Mouse_House")
            self.assertEqual(catalog.recent(), (project,))
            payload = json.loads(user.recent_projects.read_text())
            self.assertEqual(payload["paths"], [str(project.paths.root)])

    def test_catalog_refuses_unsafe_parent_before_creating_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app_root = root / "app"
            (app_root / "resources/project_template/tasks").mkdir(parents=True)
            (app_root / "resources/project_template/tasks/default.yaml").write_text(
                "task_name: Test\n"
            )
            user = UserPaths(
                config=root / "user/config",
                state=root / "user/state",
                runtime=root / "user/runtime",
                projects_parent=root / "projects",
            )
            catalog = ProjectCatalog(app=AppPaths.from_root(app_root), user=user)
            unsafe_parent = app_root / "runtime-projects"

            with self.assertRaisesRegex(ValueError, "application root"):
                catalog.create(name="Must Not Exist", parent=unsafe_parent)

            self.assertFalse(unsafe_parent.exists())

    def test_catalog_refuses_dangling_project_parent_without_creating_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app_root = root / "app"
            app_root.mkdir()
            missing_target = root / "unmounted-storage"
            dangling = root / "projects-link"
            dangling.symlink_to(missing_target, target_is_directory=True)
            user = UserPaths(
                config=root / "user/config",
                state=root / "user/state",
                runtime=root / "user/runtime",
                projects_parent=dangling,
            )
            catalog = ProjectCatalog(app=AppPaths.from_root(app_root), user=user)

            with self.assertRaisesRegex(ValueError, "dangling symbolic link"):
                catalog.ensure_projects_parent()

            self.assertFalse(missing_target.exists())

    def test_catalog_rejects_project_inside_app_before_writing_user_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app_root = root / "app"
            template = app_root / "resources/project_template/tasks/default.yaml"
            template.parent.mkdir(parents=True)
            template.write_text("task_name: Test\n", encoding="utf-8")
            app = AppPaths.from_root(app_root)
            nested = create_project(
                app_root / "invalid-project",
                name="Invalid",
                app=app,
            )
            user = UserPaths(
                config=root / "user/config",
                state=root / "user/state",
                runtime=root / "user/runtime",
                projects_parent=root / "projects",
            )
            catalog = ProjectCatalog(app=app, user=user)
            lock_before = nested.paths.ownership_lock.read_bytes()

            with self.assertRaisesRegex(ValueError, "non-overlapping"):
                catalog.open(nested.paths.root)

            self.assertFalse(user.config.exists())
            self.assertEqual(nested.paths.ownership_lock.read_bytes(), lock_before)

            user.ensure()
            payload = {
                "schema_version": 1,
                "paths": [str(nested.paths.root)],
            }
            user.recent_projects.write_text(json.dumps(payload), encoding="utf-8")
            before = user.recent_projects.read_bytes()

            self.assertEqual(catalog.recent(), ())
            self.assertEqual(user.recent_projects.read_bytes(), before)

    def test_catalog_ignores_invalid_recent_entries_without_modifying_them(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app_root = root / "app"
            app_root.mkdir()
            user = UserPaths(
                config=root / "user/config",
                state=root / "user/state",
                runtime=root / "user/runtime",
                projects_parent=root / "projects",
            )
            user.ensure()
            original = '{"schema_version":99,"paths":["/missing"]}\n'
            user.recent_projects.write_text(original)
            catalog = ProjectCatalog(app=AppPaths.from_root(app_root), user=user)

            self.assertEqual(catalog.recent(), ())
            self.assertEqual(user.recent_projects.read_text(), original)

    def test_two_projects_keep_profiles_runs_and_paths_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app_root = root / "app"
            (app_root / "resources/project_template/tasks").mkdir(parents=True)
            (app_root / "resources/project_template/tasks/default.yaml").write_text(
                "task_name: Test\n"
            )
            app = AppPaths.from_root(app_root)
            projects_parent = root / "projects"
            projects_parent.mkdir()
            first = create_project(projects_parent / "first", name="First", app=app)
            second = create_project(projects_parent / "second", name="Second", app=app)
            first_store = ProfileStore(
                first.paths.profiles,
                project_paths=first.paths,
            )
            first_store.save_experiment(ExperimentProfile("Study", "study"))
            run_dir, _ = run_context.create_run_dir(
                runs_dir=first.paths.runs,
                experiment_name="Study",
                mouse_id="Mouse",
            )

            self.assertTrue(run_dir.is_relative_to(first.paths.runs))
            self.assertEqual(len(first_store.list_experiments()), 1)
            self.assertEqual(
                ProfileStore(
                    second.paths.profiles,
                    project_paths=second.paths,
                ).list_experiments(),
                [],
            )
            with self.assertRaisesRegex(ValueError, "escapes"):
                second.paths.resolve_path(run_dir, within=second.paths.runs)

    def test_project_workflow_does_not_write_to_application_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            app_root = root / "app"
            template = app_root / "resources/project_template/tasks/default.yaml"
            template.parent.mkdir(parents=True)
            template.write_text("task_name: Test\n")
            before = {
                path.relative_to(app_root): path.read_bytes()
                for path in app_root.rglob("*")
                if path.is_file()
            }
            projects_parent = root / "projects"
            projects_parent.mkdir()

            project = create_project(
                projects_parent / "isolated",
                name="Isolated",
                app=AppPaths.from_root(app_root),
            )
            ProfileStore(
                project.paths.profiles,
                project_paths=project.paths,
            ).save_subject(
                SubjectProfile(name="Mouse", subject_id="mouse")
            )
            run_context.create_run_dir(
                runs_dir=project.paths.runs,
                experiment_name="Study",
                mouse_id="Mouse",
            )
            after = {
                path.relative_to(app_root): path.read_bytes()
                for path in app_root.rglob("*")
                if path.is_file()
            }

            self.assertEqual(after, before)

    def test_application_update_and_rollback_do_not_change_project_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            applications: list[AppPaths] = []
            for release in ("checkout-before", "checkout-after"):
                app_root = root / release
                template = app_root / "resources/project_template/tasks/default.yaml"
                template.parent.mkdir(parents=True)
                template.write_text(
                    f"task_name: {release}\n",
                    encoding="utf-8",
                )
                applications.append(AppPaths.from_root(app_root))
            projects_parent = root / "SqueakView Projects"
            projects_parent.mkdir()
            project = create_project(
                projects_parent / "MouseHouse",
                name="MouseHouse",
                app=applications[0],
            )
            ProfileStore(
                project.paths.profiles,
                project_paths=project.paths,
            ).save_experiment(ExperimentProfile("Study", "study"))
            run_context.create_run_dir(
                runs_dir=project.paths.runs,
                experiment_name="Study",
                mouse_id="Mouse",
            )
            user = UserPaths(
                config=root / "user/config",
                state=root / "user/state",
                runtime=root / "user/runtime",
                projects_parent=projects_parent,
            )
            expected = _tree_snapshot(project.paths.root)

            updated = ProjectCatalog(app=applications[1], user=user).open(
                project.paths.root
            )
            self.assertEqual(updated.metadata.project_id, project.metadata.project_id)
            self.assertEqual(_tree_snapshot(project.paths.root), expected)

            rolled_back = ProjectCatalog(app=applications[0], user=user).open(
                project.paths.root
            )
            self.assertEqual(
                rolled_back.metadata.project_id,
                project.metadata.project_id,
            )
            self.assertEqual(_tree_snapshot(project.paths.root), expected)


if __name__ == "__main__":
    unittest.main()
