use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use chrono::Utc;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanStepStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub id: usize,
    pub description: String,
    #[serde(default)]
    pub depends_on: Vec<usize>,
    #[serde(default = "default_pending_status")]
    pub status: PlanStepStatus,
    #[serde(default)]
    pub attempt_count: usize,
    #[serde(default)]
    pub last_error: Option<String>,
    #[serde(default)]
    pub started_at: Option<String>,
    #[serde(default)]
    pub completed_at: Option<String>,
    #[serde(default)]
    pub updated_at: Option<String>,
}

fn default_pending_status() -> PlanStepStatus {
    PlanStepStatus::Pending
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanArtifact {
    pub id: String,
    pub goal: String,
    pub created_at: String,
    pub steps: Vec<PlanStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalArtifact {
    pub plan_id: String,
    pub status: String,
    pub findings: Vec<String>,
    pub created_at: String,
}

pub struct PlannerEngine {
    artifact_dir: PathBuf,
}

impl PlannerEngine {
    pub fn new(artifact_dir: impl AsRef<Path>) -> Self {
        Self {
            artifact_dir: artifact_dir.as_ref().to_path_buf(),
        }
    }

    pub fn create_plan(&self, goal: &str) -> Result<PlanArtifact> {
        let id = format!("plan-{}", Utc::now().timestamp_millis());
        let steps = derive_steps(goal);
        let plan = PlanArtifact {
            id: id.clone(),
            goal: goal.to_string(),
            created_at: Utc::now().to_rfc3339(),
            steps,
        };
        validate_plan_dependencies(&plan)?;
        self.write_plan(&plan)?;
        Ok(plan)
    }

    pub fn evaluate_plan(&self, plan: &PlanArtifact) -> Result<EvalArtifact> {
        validate_plan_dependencies(plan)?;
        let mut findings = Vec::new();

        if plan.steps.is_empty() {
            findings.push("Plan has no steps".to_string());
        } else {
            findings.push(format!("Plan has {} step(s)", plan.steps.len()));
        }

        let completed = plan
            .steps
            .iter()
            .filter(|s| s.status == PlanStepStatus::Completed)
            .count();
        let in_progress = plan
            .steps
            .iter()
            .filter(|s| s.status == PlanStepStatus::InProgress)
            .count();
        let failed = plan
            .steps
            .iter()
            .filter(|s| s.status == PlanStepStatus::Failed)
            .count();
        let pending = plan
            .steps
            .iter()
            .filter(|s| s.status == PlanStepStatus::Pending)
            .count();

        findings.push(format!(
            "Step status counts: completed={completed}, in_progress={in_progress}, pending={pending}, failed={failed}"
        ));

        let ready = self.next_ready_step_for_plan(plan)?;
        if let Some(step) = ready {
            findings.push(format!("Next ready step: {}", step.id));
        } else if pending > 0 {
            findings.push("No ready step due to unresolved dependencies".to_string());
        }

        let status = if plan.steps.is_empty() {
            "reject"
        } else if failed > 0 {
            "attention"
        } else if completed == plan.steps.len() {
            "done"
        } else {
            "ready"
        };

        let eval = EvalArtifact {
            plan_id: plan.id.clone(),
            status: status.to_string(),
            findings,
            created_at: Utc::now().to_rfc3339(),
        };
        self.write_json(format!("{}-eval.json", plan.id), &eval)?;
        Ok(eval)
    }

    pub fn load_plan(&self, plan_id: &str) -> Result<PlanArtifact> {
        let path = self.plan_path(plan_id);
        let text = fs::read_to_string(&path)
            .with_context(|| format!("failed to read plan artifact: {}", path.display()))?;
        let plan: PlanArtifact = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse plan artifact: {}", path.display()))?;
        validate_plan_dependencies(&plan)?;
        Ok(plan)
    }

    pub fn mark_step_in_progress(&self, plan_id: &str, step_id: usize) -> Result<PlanArtifact> {
        let mut plan = self.load_plan(plan_id)?;
        let now = Utc::now().to_rfc3339();

        let idx = find_step_index(&plan.steps, step_id)?;
        ensure_dependencies_completed(&plan.steps, idx)?;

        let step = &mut plan.steps[idx];
        match step.status {
            PlanStepStatus::Pending | PlanStepStatus::Failed => {
                step.status = PlanStepStatus::InProgress;
                step.attempt_count += 1;
                step.started_at = Some(now.clone());
                step.updated_at = Some(now);
                step.last_error = None;
            }
            PlanStepStatus::InProgress => {}
            PlanStepStatus::Completed => bail!("step {step_id} is already completed"),
        }

        self.write_plan(&plan)?;
        Ok(plan)
    }

    pub fn mark_step_completed(&self, plan_id: &str, step_id: usize) -> Result<PlanArtifact> {
        let mut plan = self.load_plan(plan_id)?;
        let now = Utc::now().to_rfc3339();

        let idx = find_step_index(&plan.steps, step_id)?;
        ensure_dependencies_completed(&plan.steps, idx)?;
        let step = &mut plan.steps[idx];

        if step.status == PlanStepStatus::Completed {
            return Ok(plan);
        }

        if step.status == PlanStepStatus::Pending {
            step.attempt_count += 1;
            step.started_at = Some(now.clone());
        }

        step.status = PlanStepStatus::Completed;
        step.completed_at = Some(now.clone());
        step.updated_at = Some(now);
        step.last_error = None;

        self.write_plan(&plan)?;
        Ok(plan)
    }

    pub fn mark_step_failed(
        &self,
        plan_id: &str,
        step_id: usize,
        reason: &str,
    ) -> Result<PlanArtifact> {
        let mut plan = self.load_plan(plan_id)?;
        let now = Utc::now().to_rfc3339();

        let idx = find_step_index(&plan.steps, step_id)?;
        let step = &mut plan.steps[idx];
        if step.status == PlanStepStatus::Completed {
            bail!("step {step_id} is already completed");
        }

        if step.status == PlanStepStatus::Pending {
            step.attempt_count += 1;
            step.started_at = Some(now.clone());
        }

        step.status = PlanStepStatus::Failed;
        step.last_error = Some(reason.to_string());
        step.updated_at = Some(now);

        self.write_plan(&plan)?;
        Ok(plan)
    }

    pub fn resume_plan(&self, plan_id: &str) -> Result<Option<PlanStep>> {
        let plan = self.load_plan(plan_id)?;
        self.next_ready_step_for_plan(&plan)
    }

    fn next_ready_step_for_plan(&self, plan: &PlanArtifact) -> Result<Option<PlanStep>> {
        validate_plan_dependencies(plan)?;

        let step_by_id: HashMap<usize, &PlanStep> = plan.steps.iter().map(|s| (s.id, s)).collect();

        for step in &plan.steps {
            if !matches!(
                step.status,
                PlanStepStatus::Pending | PlanStepStatus::Failed
            ) {
                continue;
            }
            let deps_done = step.depends_on.iter().all(|dep_id| {
                step_by_id
                    .get(dep_id)
                    .map(|dep| dep.status == PlanStepStatus::Completed)
                    .unwrap_or(false)
            });
            if deps_done {
                return Ok(Some(step.clone()));
            }
        }

        Ok(None)
    }

    fn write_plan(&self, plan: &PlanArtifact) -> Result<()> {
        self.write_json(format!("{}.json", plan.id), plan)
    }

    fn plan_path(&self, plan_id: &str) -> PathBuf {
        self.artifact_dir.join(format!("{plan_id}.json"))
    }

    fn write_json<T: Serialize>(&self, filename: String, payload: &T) -> Result<()> {
        fs::create_dir_all(&self.artifact_dir)?;
        let path = self.artifact_dir.join(filename);
        fs::write(path, serde_json::to_string_pretty(payload)?)?;
        Ok(())
    }
}

fn derive_steps(goal: &str) -> Vec<PlanStep> {
    let pieces: Vec<String> = goal
        .split(" and ")
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect();

    let raw_steps = if pieces.is_empty() {
        vec![
            "Clarify objective".to_string(),
            "Implement minimal slice".to_string(),
        ]
    } else {
        pieces
    };

    raw_steps
        .into_iter()
        .enumerate()
        .map(|(idx, description)| {
            let id = idx + 1;
            PlanStep {
                id,
                description,
                depends_on: if id == 1 { Vec::new() } else { vec![id - 1] },
                status: PlanStepStatus::Pending,
                attempt_count: 0,
                last_error: None,
                started_at: None,
                completed_at: None,
                updated_at: None,
            }
        })
        .collect()
}

fn validate_plan_dependencies(plan: &PlanArtifact) -> Result<()> {
    let ids: HashSet<usize> = plan.steps.iter().map(|s| s.id).collect();
    for step in &plan.steps {
        if step.depends_on.contains(&step.id) {
            bail!("step {} cannot depend on itself", step.id);
        }
        for dep in &step.depends_on {
            if !ids.contains(dep) {
                bail!("step {} depends on missing step {}", step.id, dep);
            }
        }
    }

    let graph: HashMap<usize, Vec<usize>> = plan
        .steps
        .iter()
        .map(|s| (s.id, s.depends_on.clone()))
        .collect();

    let mut visiting = HashSet::new();
    let mut visited = HashSet::new();
    for id in &ids {
        detect_cycle(*id, &graph, &mut visiting, &mut visited)?;
    }
    Ok(())
}

fn detect_cycle(
    id: usize,
    graph: &HashMap<usize, Vec<usize>>,
    visiting: &mut HashSet<usize>,
    visited: &mut HashSet<usize>,
) -> Result<()> {
    if visited.contains(&id) {
        return Ok(());
    }
    if !visiting.insert(id) {
        bail!("dependency cycle detected at step {id}");
    }

    if let Some(deps) = graph.get(&id) {
        for dep in deps {
            detect_cycle(*dep, graph, visiting, visited)?;
        }
    }

    visiting.remove(&id);
    visited.insert(id);
    Ok(())
}

fn find_step_index(steps: &[PlanStep], step_id: usize) -> Result<usize> {
    steps
        .iter()
        .position(|step| step.id == step_id)
        .ok_or_else(|| anyhow::anyhow!("step {step_id} does not exist"))
}

fn ensure_dependencies_completed(steps: &[PlanStep], index: usize) -> Result<()> {
    let step = &steps[index];
    let map: HashMap<usize, &PlanStep> = steps.iter().map(|s| (s.id, s)).collect();

    for dep in &step.depends_on {
        let dep_step = map
            .get(dep)
            .ok_or_else(|| anyhow::anyhow!("dependency step {dep} missing"))?;
        if dep_step.status != PlanStepStatus::Completed {
            bail!(
                "step {} is blocked by dependency {} (status={:?})",
                step.id,
                dep,
                dep_step.status
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{PlanStepStatus, PlannerEngine};

    fn temp_test_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "agent_in_rust_scratch_{name}_{}_{}",
            std::process::id(),
            nanos
        ));
        fs::create_dir_all(&dir).expect("failed to create temp test directory");
        dir
    }

    #[test]
    fn planner_writes_plan_and_eval_artifacts() {
        let dir = temp_test_dir("planner_artifacts");
        let planner = PlannerEngine::new(&dir);

        let plan = planner
            .create_plan("add scheduler and policy engine")
            .expect("plan creation should succeed");
        let eval = planner
            .evaluate_plan(&plan)
            .expect("evaluation should succeed");

        assert_eq!(eval.plan_id, plan.id);
        assert_eq!(eval.status, "ready");

        let plan_file = dir.join(format!("{}.json", plan.id));
        let eval_file = dir.join(format!("{}-eval.json", plan.id));
        assert!(plan_file.exists());
        assert!(eval_file.exists());

        fs::remove_dir_all(dir).expect("failed to cleanup temp directory");
    }

    #[test]
    fn resume_returns_next_unblocked_step() {
        let dir = temp_test_dir("planner_resume");
        let planner = PlannerEngine::new(&dir);
        let plan = planner
            .create_plan("first and second and third")
            .expect("plan should be created");

        let first = planner
            .resume_plan(&plan.id)
            .expect("resume should succeed")
            .expect("first step should be ready");
        assert_eq!(first.id, 1);

        planner
            .mark_step_completed(&plan.id, 1)
            .expect("step 1 complete should succeed");

        let second = planner
            .resume_plan(&plan.id)
            .expect("resume should succeed")
            .expect("second step should be ready");
        assert_eq!(second.id, 2);

        fs::remove_dir_all(dir).expect("failed to cleanup temp directory");
    }

    #[test]
    fn start_blocked_step_fails_until_dependency_completed() {
        let dir = temp_test_dir("planner_blocked");
        let planner = PlannerEngine::new(&dir);
        let plan = planner
            .create_plan("first and second")
            .expect("plan should be created");

        let err = planner
            .mark_step_in_progress(&plan.id, 2)
            .expect_err("step 2 should be blocked");
        assert!(format!("{err:#}").contains("blocked by dependency"));

        planner
            .mark_step_completed(&plan.id, 1)
            .expect("step 1 complete should succeed");
        let updated = planner
            .mark_step_in_progress(&plan.id, 2)
            .expect("step 2 start should succeed after dep complete");
        assert_eq!(updated.steps[1].status, PlanStepStatus::InProgress);

        fs::remove_dir_all(dir).expect("failed to cleanup temp directory");
    }

    #[test]
    fn status_updates_persist_across_reload() {
        let dir = temp_test_dir("planner_persist");
        let planner = PlannerEngine::new(&dir);
        let plan = planner
            .create_plan("first and second")
            .expect("plan should be created");

        planner
            .mark_step_in_progress(&plan.id, 1)
            .expect("step should start");
        planner
            .mark_step_failed(&plan.id, 1, "boom")
            .expect("step should fail");

        let reloaded = planner.load_plan(&plan.id).expect("reload should succeed");
        assert_eq!(reloaded.steps[0].status, PlanStepStatus::Failed);
        assert_eq!(reloaded.steps[0].last_error.as_deref(), Some("boom"));

        fs::remove_dir_all(dir).expect("failed to cleanup temp directory");
    }
}
