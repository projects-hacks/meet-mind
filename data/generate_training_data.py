"""Generate diverse, production-grade training data for FunctionGemma fine-tuning.

CRITICAL: Training data format MUST match Analyst._build_prompt() exactly.
This prevents format-based overfitting and ensures real-world generalization.

The real prompt looks like:
    Meeting: {topic} | Domain: {domain}
    Key points: {joined key_points}
    Whiteboard: {content}
    Recent timeline: {timeline entries}
    Tracked: {N} actions, {M} decisions, {K} gaps
    Recent agent actions: {actions}
    Consecutive observations without action: {count}

    What is the single most important action to take now?

Meeting topics MUST be generic (not action-leaking). The model should learn
to decide actions based on KEY POINTS and TIMELINE content, not topic titles.

Usage:
    python3 data/generate_training_data.py
"""

import json
import os
import random

random.seed(42)

OUTPUT = os.path.join(os.path.dirname(__file__), "analyst_training.jsonl")

SYSTEM_MSG = (
    "You are a meeting analyst. Based on the meeting state, decide the single "
    "most important action. Respond with JSON: "
    '{"action":"name","params":{...},"reasoning":"why"}'
)

# ── Building blocks for diverse generation ──

NAMES = [
    "Sarah", "Mike", "Lisa", "Tom", "Alex", "Rachel", "Dave", "Emily",
    "James", "Karen", "Dan", "Maria", "John", "Priya", "Carlos",
    "Wei", "Aisha", "Yuki", "Olga", "Raj", "Nina", "Ben", "Chen",
]

DEADLINES = [
    "Friday", "next Monday", "end of week", "tomorrow", "end of day",
    "next Wednesday", "by Thursday", "in 2 weeks", "end of month",
    "before the board meeting", "before launch", "ASAP", "this sprint",
]

PRIORITIES = ["high", "medium", "low"]

# Generic meeting topics — these do NOT leak the action type.
# The model should NOT be able to guess "provide_insight" just because
# the topic says "Security Risk". Use normal meeting names.
GENERIC_TOPICS = [
    "Weekly Standup", "Sprint Planning", "Team Sync", "Design Review",
    "Quarterly Review", "Strategy Meeting", "All Hands", "1:1 Catch-up",
    "Client Call", "Architecture Review", "Planning Session", "Project Update",
    "Product Review", "Pipeline Review", "Retrospective", "Team Meeting",
    "Status Update", "Kickoff", "Brainstorm Session", "Working Session",
    "Demo Day", "Roadmap Review", "Budget Review", "Cross-Team Sync",
    "Leadership Sync", "Stand-Up", "Weekly Review", "Check-In",
    "Technical Discussion", "Stakeholder Update",
]


# ── Prompt builder: mirrors Analyst._build_prompt() exactly ──

def build_prompt(
    topic: str | None,
    domain: str,
    key_points: list[str],
    *,
    whiteboard: str = "",
    timeline: list[str] | None = None,
    tracked_actions: int | None = None,
    tracked_decisions: int | None = None,
    tracked_gaps: int | None = None,
    recent_agent_actions: list[str] | None = None,
    consecutive_observe: int | None = None,
) -> str:
    """Build a prompt that matches the real Analyst._build_prompt() format.

    This ensures training data looks EXACTLY like what the model sees in
    production, preventing format-based overfitting.
    """
    # Use a generic topic if none provided
    if topic is None:
        topic = random.choice(GENERIC_TOPICS)

    parts = [
        f"Meeting: {topic} | Domain: {domain}",
        f"Key points: {'; '.join(key_points)}",
    ]

    # Whiteboard — sometimes present
    if whiteboard:
        parts.append(f"Whiteboard: {whiteboard}")
    elif random.random() < 0.2:
        parts.append("Whiteboard: none")

    # Recent timeline — mimics real conversation snippets
    if timeline:
        parts.append(f"Recent timeline: {'; '.join(timeline)}")
    elif random.random() < 0.2:
        parts.append("Recent timeline: none")

    # Tracked counts — randomize if not explicit
    ta = tracked_actions if tracked_actions is not None else random.randint(0, 3)
    td = tracked_decisions if tracked_decisions is not None else random.randint(0, 2)
    tg = tracked_gaps if tracked_gaps is not None else random.randint(0, 1)
    parts.append(f"Tracked: {ta} actions, {td} decisions, {tg} gaps")

    # Recent agent actions — randomize
    if recent_agent_actions:
        parts.append(f"Recent agent actions: {', '.join(recent_agent_actions)}")
    elif random.random() < 0.3:
        past = random.sample(
            ["continue_observing", "extract_action_item", "log_decision"],
            k=random.randint(1, 2),
        )
        parts.append(f"Recent agent actions: {', '.join(past)}")
    else:
        parts.append("Recent agent actions: none")

    # Consecutive observe count
    co = consecutive_observe if consecutive_observe is not None else 0
    parts.append(f"Consecutive observations without action: {co}")

    # Final question — always present in real prompt
    parts.append("")
    parts.append("What is the single most important action to take now?")

    return "\n".join(parts)


def make_msg(user_content: str, assistant_content: str):
    """Build a training message triple (system, user, assistant)."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


# ════════════════════════════════════════════════════════════════════
# 1. extract_action_item
# ════════════════════════════════════════════════════════════════════

def gen_extract_action_items():
    """Generate extract_action_item examples — diverse scenarios."""
    examples = []

    scenarios = [
        # (domain, key_points_templates, timeline_templates, task, base_priority)
        ("engineering",
         ["{name} volunteered to refactor the auth module", "Deadline agreed as {deadline}"],
         ["{name}: I can handle the auth refactoring", "Manager: Great, get it done by {deadline}"],
         "Refactor auth module", "high"),
        ("engineering",
         ["Need GitHub Actions pipeline", "Docker builds are manual right now"],
         ["{name}: I will set up CI/CD", "Lead: Get it done by {deadline}"],
         "Set up CI/CD pipeline with GitHub Actions", "high"),
        ("engineering",
         ["Critical bug in payment processing", "Users seeing failed transactions"],
         ["{name}: This is P0, I will fix the payment bug", "Manager: Need it by {deadline}"],
         "Fix critical payment processing bug", "high"),
        ("engineering",
         ["API docs are outdated", "New endpoints not documented"],
         ["{name}: I will update the docs", "Lead: Have it by {deadline}"],
         "Update API documentation for new endpoints", "medium"),
        ("engineering",
         ["Unit test coverage at 45 percent", "Need 80 percent before release"],
         ["{name}: I will write tests for the user service", "Target: {deadline}"],
         "Write unit tests for user service", "medium"),
        ("engineering",
         ["PR backlog growing", "12 PRs pending review"],
         ["{name}: I will clear the backlog by {deadline}"],
         "Review and clear pending PR backlog", "medium"),
        ("engineering",
         ["API response time 3 seconds", "Target is under 500ms"],
         ["{name}: I will profile and optimize the slow endpoints by {deadline}"],
         "Profile and optimize slow API endpoints", "high"),
        ("engineering",
         ["Need to add user preferences table", "Schema change required"],
         ["{name}: I will prepare the migration script", "Due: {deadline}"],
         "Prepare database migration script for user preferences", "medium"),
        # Sales
        ("sales",
         ["Client wants custom pricing", "Enterprise tier discussion ongoing"],
         ["{name}: I will draft the enterprise proposal for Acme", "Send by {deadline}"],
         "Draft enterprise proposal for Acme Corp", "high"),
        ("sales",
         ["Demo went well with client", "Client had questions about security"],
         ["{name}: I will schedule the follow-up call by {deadline}"],
         "Schedule follow-up call with client regarding security questions", "high"),
        ("sales",
         ["New competitor entered market", "Undercutting our prices by 30 percent"],
         ["{name}: I will prepare a competitive analysis by {deadline}"],
         "Prepare competitive analysis of new market entrant", "high"),
        ("sales",
         ["Deal pipeline needs updating", "Several stage changes since last week"],
         ["{name}: I will update the CRM by {deadline}"],
         "Update CRM with current deal stage changes", "low"),
        # Operations
        ("operations",
         ["Need backup supplier", "Current vendor unreliable last 3 months"],
         ["{name}: I will contact three backup vendors by {deadline}"],
         "Contact backup vendors and obtain quotes", "high"),
        ("operations",
         ["Onboarding process is tribal knowledge", "Need written SOP"],
         ["{name}: I will document the onboarding process by {deadline}"],
         "Document employee onboarding SOP", "medium"),
        ("operations",
         ["Discrepancies in warehouse counts", "Need physical audit"],
         ["{name}: I will do the inventory audit by {deadline}"],
         "Conduct physical warehouse inventory audit", "medium"),
        # Product
        ("product",
         ["Need feedback on new feature", "Target 15 user interviews"],
         ["{name}: I will schedule and conduct interviews by {deadline}"],
         "Conduct 15 user interviews on new feature", "high"),
        ("product",
         ["New dashboard design needed", "Stakeholder review next week"],
         ["{name}: I will create wireframes by {deadline}"],
         "Create dashboard wireframes for stakeholder review", "medium"),
        ("product",
         ["Need event tracking for new feature", "Cannot measure success without it"],
         ["{name}: I will implement analytics tracking by {deadline}"],
         "Implement analytics event tracking for new feature", "medium"),
        ("product",
         ["v3.0 launching next week", "Marketing needs release notes"],
         ["{name}: I will write the release notes by {deadline}"],
         "Write v3.0 release notes for marketing", "medium"),
    ]

    for domain, pt, tl, task, priority in scenarios:
        name = random.choice(NAMES)
        deadline = random.choice(DEADLINES)

        key_points = [p.format(name=name, deadline=deadline) for p in pt]
        timeline = [t.format(name=name, deadline=deadline) for t in tl]

        user = build_prompt(None, domain, key_points, timeline=timeline)
        assistant = json.dumps({
            "action": "extract_action_item",
            "params": {"owner": name, "task": task, "deadline": deadline, "priority": priority},
            "reasoning": f"{name} was assigned '{task}' with {deadline} deadline",
        })
        examples.append(make_msg(user, assistant))

    return examples


# ════════════════════════════════════════════════════════════════════
# 2. log_decision
# ════════════════════════════════════════════════════════════════════

def gen_log_decisions():
    """Generate log_decision examples."""
    examples = []

    scenarios = [
        # (domain, decision, rejected, rationale, timeline_snippet)
        ("engineering", "Use React for frontend", ["Vue", "Angular", "Svelte"],
         "Largest ecosystem and hiring pool",
         "Lead: Everyone agrees? OK lets go with React"),
        ("engineering", "Use PostgreSQL as primary database", ["MongoDB", "DynamoDB"],
         "ACID compliance without vendor lock-in",
         "Lead: Everyone agrees? OK use PostgreSQL"),
        ("engineering", "Use REST API over GraphQL", ["GraphQL", "gRPC"],
         "Simpler for current use case and team expertise",
         "CTO: After considering the tradeoffs use REST"),
        ("engineering", "Deploy on Google Cloud Platform", ["AWS", "Azure"],
         "Better AI services and cost structure",
         "Lead: I think GCP is the right call"),
        ("engineering", "Use Kafka for event streaming", ["RabbitMQ", "Redis Streams", "SQS"],
         "Better throughput for real-time analytics",
         "Architect: Everyone agrees? OK use Kafka"),
        ("engineering", "Use pytest with coverage enforcement", ["unittest", "nose2"],
         "Better plugin ecosystem and fixture system",
         "Lead: Lets go with pytest"),
        ("engineering", "Use Kubernetes for deployment", ["ECS", "Docker Swarm", "Bare EC2"],
         "Team has K8s experience and need auto-scaling",
         "CTO: Kubernetes it is"),
        ("engineering", "Implement OAuth2 with PKCE", ["Session-based auth", "API keys only"],
         "Industry standard and more secure for SPAs",
         "Security: OAuth2 with PKCE is the right choice"),
        # Sales
        ("sales", "Offer 15% volume discount to Acme Corp", ["20% discount", "No discount"],
         "Revisit if they commit to annual contract",
         "VP Sales: After considering everything 15 percent for Acme"),
        ("sales", "Require 2-year minimum commitment for enterprise",
         ["1-year terms", "Month-to-month"],
         "Reduces churn risk at lower price point",
         "Sales Lead: 2-year commitment for enterprise"),
        ("sales", "Split West Coast between Lisa and Tom",
         ["Single rep coverage", "Three-way split"],
         "Two reps can cover more effectively",
         "VP: OK split the West Coast between Lisa and Tom"),
        ("sales", "Switch to usage-based pricing for API product",
         ["Flat monthly fee", "Per-seat pricing"],
         "Aligns cost with customer value and reduces entry barrier",
         "VP: Everyone agrees on usage-based pricing"),
        # Operations
        ("operations", "Automate QA pipeline except edge cases",
         ["Fully manual", "Full automation"],
         "Balances speed with quality for production",
         "Ops Lead: Automate QA except edge cases"),
        ("operations", "Use Google Cloud for infrastructure",
         ["AWS", "Azure", "On-premise"],
         "Best AI integration and competitive pricing",
         "CTO: Google Cloud for infrastructure"),
        ("operations", "Move to 4-day work week for warehouse",
         ["Standard 5-day", "3 rotating shifts"],
         "Improves retention and reduces overtime costs",
         "Director: OK 4-day week for warehouse"),
        # Product
        ("product", "Prioritize dark mode for next sprint",
         ["Export to PDF", "Mobile app"],
         "Most requested feature by users based on NPS surveys",
         "PM: Everyone agrees? OK dark mode next sprint"),
        ("product", "Launch to existing customers first then expand",
         ["Big public launch", "Invite-only beta"],
         "Lower risk and generates testimonials",
         "VP Product: Launch to existing customers first"),
        ("product", "Enterprise-first strategy starting Q3",
         ["SMB focus", "Hybrid approach"],
         "Enterprise shows 3x more profitability",
         "CEO: After considering everything enterprise first starting Q3"),
        # General
        ("general", "Switch to bi-weekly all-hands instead of weekly",
         ["Weekly all-hands", "Monthly all-hands"],
         "Weekly is too frequent, monthly too infrequent",
         "CEO: OK bi-weekly all-hands from now on"),
    ]

    for domain, decision, rejected, rationale, tl_snippet in scenarios:
        key_points = ["Discussed options", "Team evaluated alternatives"]
        timeline = [tl_snippet]

        user = build_prompt(None, domain, key_points, timeline=timeline)
        assistant = json.dumps({
            "action": "log_decision",
            "params": {"decision": decision, "alternatives_rejected": rejected, "rationale": rationale},
            "reasoning": f"Firm decision made: {decision}",
        })
        examples.append(make_msg(user, assistant))

    return examples


# ════════════════════════════════════════════════════════════════════
# 3. flag_gap
# ════════════════════════════════════════════════════════════════════

def gen_flag_gaps():
    """Generate flag_gap examples — identifying unresolved issues."""
    examples = []

    scenarios = [
        # (domain, key_points, gap_topic, gap_type, suggestion, timeline_snippet)
        ("engineering",
         ["Database migration discussed but no owner volunteered"],
         "Database migration", "no_owner",
         "Assign a database migration lead before next standup",
         "Someone: We should deal with the migration. Nobody: ... (silence)"),
        ("engineering",
         ["CI pipeline is broken but nobody assigned to fix it"],
         "CI Pipeline Fix", "no_owner",
         "Assign someone to fix CI pipeline before next merge",
         "Lead: CI is broken, who can fix it? Silence"),
        ("engineering",
         ["K8s vs ECS debate tabled without resolution"],
         "Deployment platform selection", "no_decision",
         "Schedule follow-up meeting to decide deployment platform",
         "CTO: We need to decide but lets move on for now"),
        ("engineering",
         ["Microservices discussed but service boundaries not defined"],
         "Microservices migration scope", "unclear_scope",
         "Map service boundaries before starting migration work",
         "Lead: Microservices are important but scope is unclear"),
        ("engineering",
         ["Security audit needed before launch but nobody owns it"],
         "Security audit", "no_owner",
         "Urgently assign owner for security audit",
         "Manager: Pen test is important but who is doing it?"),
        ("engineering",
         ["API optimization discussed but no timeline set"],
         "Performance optimization timeline", "no_deadline",
         "Set deadline for API performance improvements",
         "CTO: Performance is a problem but no deadline discussed"),
        ("engineering",
         ["Discussed code quality but no specific actions"],
         "Code quality investigation", "no_owner",
         "Clarify what needs improvement and assign a specific owner",
         "Someone: Should probably look into that at some point"),
        ("engineering",
         ["Security testing mentioned but no date committed"],
         "Penetration testing schedule", "no_deadline",
         "Schedule pen testing before launch date",
         "Lead: We need pen testing but no date set"),
        # Sales
        ("sales",
         ["Client complaint needs resolution but account manager is on leave"],
         "Client escalation coverage", "no_owner",
         "Assign temporary account manager for escalated client",
         "VP: Who is handling the escalation? Nobody answered"),
        ("sales",
         ["Pricing restructure discussed but no conclusion reached"],
         "Pricing restructure decision", "no_decision",
         "Make pricing decision before end of quarter",
         "VP: Pricing is important but we did not resolve it"),
        ("sales",
         ["Competitor analysis needed but no timeline set"],
         "Competitive analysis timeline", "no_deadline",
         "Set deadline for competitive pricing analysis",
         "VP: We should analyze competitors but when?"),
        ("sales",
         ["International expansion discussed but target markets unclear"],
         "International expansion scope", "unclear_scope",
         "Research and shortlist 2-3 European markets to enter first",
         "CEO: Europe is important but which countries?"),
        # Operations
        ("operations",
         ["150K unallocated budget with no spending plan"],
         "Budget allocation", "no_decision",
         "Prepare allocation options for next budget meeting",
         "CFO: Who is allocating the remaining budget? Silence"),
        ("operations",
         ["Packaging cost comparison needed but no due date"],
         "Cost analysis deadline", "no_deadline",
         "Set deadline for packaging cost analysis",
         "Manager: We need cost analysis but when?"),
        ("operations",
         ["New hires need training but no trainer assigned"],
         "Training program ownership", "no_owner",
         "Designate training coordinator for new hire onboarding",
         "HR: Training is important but who runs it?"),
        ("operations",
         ["Process automation mentioned but which processes is unclear"],
         "Automation initiative scope", "unclear_scope",
         "Identify and prioritize processes for automation",
         "Director: Automation is a priority but what exactly?"),
        # Product
        ("product",
         ["Email campaign is important but no owner assigned"],
         "Email campaign ownership", "no_owner",
         "Assign someone for email campaign before feature launch",
         "PM: Blog and social covered but who does email?"),
        ("product",
         ["User interviews agreed upon but no dates planned"],
         "User research scheduling", "no_deadline",
         "Schedule user interview dates within next 2 weeks",
         "PM: We agreed on interviews but no dates set"),
        ("product",
         ["Three design approaches discussed but team could not agree"],
         "Feature scope decision", "no_decision",
         "Use data to break deadlock: run quick user poll on approaches",
         "PM: Feature scope is important but we could not agree"),
        ("product",
         ["Platform redesign suggested but requirements are missing"],
         "Platform redesign requirements", "unclear_scope",
         "Gather requirements and define MVP scope for platform redesign",
         "PM: Redesign needed but scope is completely undefined"),
        ("engineering",
         ["Hiring priority debated: senior engineer vs DevOps"],
         "Hiring priority", "no_decision",
         "Define hiring priority based on current project needs",
         "CTO: We cannot agree on who to hire first"),
    ]

    for domain, kp, gap_topic, gap_type, suggestion, tl in scenarios:
        user = build_prompt(None, domain, kp, timeline=[tl])
        assistant = json.dumps({
            "action": "flag_gap",
            "params": {"topic": gap_topic, "gap_type": gap_type, "suggestion": suggestion},
            "reasoning": kp[0],
        })
        examples.append(make_msg(user, assistant))

    return examples


# ════════════════════════════════════════════════════════════════════
# 4. request_artifact
# ════════════════════════════════════════════════════════════════════

def gen_request_artifacts():
    """Generate request_artifact examples — triggered by wrapping-up or request."""
    examples = []

    scenarios = [
        # (domain, key_points, artifact_type, context_summary, timeline, whiteboard)
        ("engineering",
         ["Full system architecture discussed", "Tech stack and deployment decided"],
         "architecture_doc",
         "Microservices: API Gateway, Auth, User DB, Cache. FastAPI + PostgreSQL + Redis. K8s on GCP.",
         ["Architect: We should capture this while it is fresh"],
         "API → Auth → DB pipeline; Redis cache layer"),
        ("engineering",
         ["Sprint goals with tasks, owners, and dependencies"],
         "impl_plan",
         "Sprint 14: Auth refactor (Sarah), DB migration (Mike), CI/CD (Tom). 2-week sprint.",
         ["PM: Can someone put together a formal document?"],
         "Sprint board snapshot with task owners"),
        ("engineering",
         ["Data pipeline design with streaming and analytics"],
         "architecture_doc",
         "Kafka → Spark processing → PostgreSQL + S3. Real-time dashboard via WebSocket.",
         ["Lead: Can someone document this discussion?"],
         "Data flow diagram: ingestion → processing → storage"),
        ("engineering",
         ["6-month migration plan with milestones and risks"],
         "impl_plan",
         "Monolith to microservices: Phase 1 extract auth, Phase 2 extract payments. 6 months.",
         ["CTO: Let us document this before we forget the details"],
         "Migration timeline with risk markers"),
        ("engineering",
         ["15+ points discussed in 90-minute meeting", "Meeting wrapping up"],
         "meeting_summary",
         "Extended planning session: 5 action items, 4 decisions, 2 gaps.",
         ["Lead: OK we have been going for a while. Any last things?"],
         ""),
        ("engineering",
         ["8 key points logged, multiple discussions complete"],
         "meeting_summary",
         "Sprint planning complete: 3 action items assigned, 2 key decisions made, 1 gap flagged.",
         ["PM: I think we have covered everything"],
         ""),
        # Sales
        ("sales",
         ["Complete deal context including pricing and competitive positioning"],
         "sales_brief",
         "Acme Corp: $50K deal, 15% discount, Q2 close. Competitive positioning vs FooBar.",
         ["VP: Let us document this while its fresh"],
         "Deal terms and competitive landscape"),
        ("sales",
         ["Partnership terms and integration plan discussed"],
         "sales_brief",
         "Partner API program: 20% rev share, sandbox access, co-marketing. Launch Q3.",
         ["VP: Let us document this before we forget"],
         "Partnership terms matrix"),
        ("sales",
         ["Full quarter performance reviewed"],
         "meeting_summary",
         "Q2: 92% of target, 3 deals slipped. Strong enterprise pipeline. Focus on retention.",
         ["VP: I think we have enough to write this up"],
         "Q2 performance dashboard"),
        # Product
        ("product",
         ["Product roadmap with features, timelines, resources"],
         "impl_plan",
         "Q3: Dark mode (2 weeks), Export (3 weeks), API v2 (4 weeks). 2 engineers + 1 designer.",
         ["PM: I think we have enough to write this up"],
         "Roadmap timeline with resource allocation"),
        ("product",
         ["Key metrics, product updates, and strategic decisions"],
         "meeting_summary",
         "MAU up 40%, NPS at 72, 3 new features shipped. Pivoting enterprise-first Q3.",
         ["CEO: Can someone put together a formal document?"],
         ""),
        # Operations
        ("operations",
         ["New employee onboarding steps with responsible parties"],
         "process_spec",
         "Day 1: IT setup + HR. Week 1: Team intro + tools. Month 1: First project + mentor.",
         ["HR: We should capture this while it is fresh"],
         "Onboarding checklist by week"),
        ("operations",
         ["End-to-end warehouse workflow with quality gates"],
         "process_spec",
         "Picking → Packing → QA check → Labeling → Shipping. 3 quality gates. 200 orders/day.",
         ["Ops: Can someone document this process?"],
         "Workflow diagram with quality checkpoints"),
        # General
        ("general",
         ["Company-wide updates, announcements, and Q&A highlights"],
         "meeting_summary",
         "New CTO announcement, Q2 results exceeded target, office move in September.",
         ["CEO: Let us document this before we forget the details"],
         ""),
    ]

    for domain, kp, atype, ctx, tl, wb in scenarios:
        user = build_prompt(
            None, domain, kp,
            whiteboard=wb if wb else "",
            timeline=tl,
            tracked_actions=random.randint(2, 5),
            tracked_decisions=random.randint(1, 3),
            tracked_gaps=random.randint(0, 2),
        )
        assistant = json.dumps({
            "action": "request_artifact",
            "params": {"artifact_type": atype, "context_summary": ctx, "domain": domain},
            "reasoning": f"Team has enough context to generate {atype}",
        })
        examples.append(make_msg(user, assistant))

    return examples


# ════════════════════════════════════════════════════════════════════
# 5. suggest_next_step
# ════════════════════════════════════════════════════════════════════

def gen_suggest_next_steps():
    """Generate suggest_next_step examples — proactive advisor behavior.

    KEY: These are triggered when the agent has been observing for a while
    (consecutive_observe >= 2) and notices a gap in the discussion.
    """
    examples = []

    scenarios = [
        # (domain, key_points, suggestion, reason)
        ("engineering",
         ["Discussed API design for 20 minutes", "No mention of deployment or scaling"],
         "Discuss deployment strategy and scaling requirements for the API",
         "Architecture decisions without deployment context may need rework later"),
        ("engineering",
         ["Sprint tasks assigned", "Dependencies between tasks not discussed"],
         "Map task dependencies to avoid blocking. Auth must complete before API changes.",
         "Parallel work without dependency mapping leads to merge conflicts"),
        ("engineering",
         ["Root cause identified for incident", "No discussion of prevention"],
         "Add monitoring and alerting for the failure mode that caused this incident",
         "Without prevention measures, the same incident will likely recur"),
        ("engineering",
         ["Discussed hiring 3 people", "Interview process not planned"],
         "Define interview process, evaluation criteria, and target hiring timeline",
         "Posting jobs without a clear process causes delays and inconsistent hiring"),
        ("engineering",
         ["New features discussed for 30 minutes", "Security implications not mentioned"],
         "Review security implications of the new features before implementation",
         "Security issues found late are 10x more expensive to fix"),
        ("engineering",
         ["Test coverage discussed", "No mention of test environments or staging"],
         "Set up dedicated staging environment for integration tests",
         "Testing in production-like environments catches environment-specific issues"),
        ("engineering",
         ["Retrospective: discussed what went wrong", "No concrete improvement actions"],
         "Convert each retrospective insight into a specific action with owner and deadline",
         "Retrospectives without concrete actions lead to the same problems repeating"),
        ("engineering",
         ["Data model changes discussed", "Backward compatibility not mentioned"],
         "Assess backward compatibility impact of data model changes on existing clients",
         "Breaking changes without versioning plan causes client integration failures"),
        ("engineering",
         ["Feature freeze date set", "No rollback plan discussed"],
         "Define rollback plan and feature flag strategy before the release date",
         "Releases without rollback plans risk extended outages if issues are found"),
        # Product
        ("product",
         ["New feature discussed", "No success metrics or KPIs defined"],
         "Define success metrics and KPIs for the new feature before development starts",
         "Without measurable goals, team cannot evaluate if the feature was worth building"),
        ("product",
         ["Roadmap discussed", "Customer feedback data not referenced"],
         "Cross-reference roadmap priorities with recent customer feedback and support tickets",
         "Roadmap aligned with customer needs has higher adoption"),
        ("product",
         ["Support tickets reviewed", "Root causes not categorized"],
         "Categorize top support issues by root cause and prioritize fixes by frequency",
         "Fixing root causes reduces ticket volume more effectively"),
        ("product",
         ["Feature list and marketing plan discussed", "Support scaling not mentioned"],
         "Discuss customer support scaling plan for launch",
         "Post-launch support failures damage brand and customer trust"),
        # Sales
        ("sales",
         ["Reviewed all deals", "Win/loss patterns not discussed"],
         "Analyze win/loss patterns to identify what is working and where deals stall",
         "Pattern analysis helps focus effort on highest-impact activities"),
        ("sales",
         ["Pricing discussed", "Customer willingness to pay not considered"],
         "Conduct price sensitivity analysis before committing to new pricing tiers",
         "Pricing without market validation risks revenue loss or churn"),
        ("sales",
         ["Onboarding process discussed", "No success manager or milestones"],
         "Assign customer success manager and define 30-60-90 day milestones",
         "Structured onboarding with milestones reduces early-stage churn"),
        ("sales",
         ["Market trends discussed for 15 minutes", "Not connected to product roadmap"],
         "Schedule joint session with product team to align market insights with roadmap",
         "Market intelligence is wasted if it does not influence product decisions"),
        # Operations
        ("operations",
         ["Reviewed current workflow", "Bottlenecks and throughput not discussed"],
         "Identify biggest bottleneck in workflow and measure throughput at each step",
         "Optimization without bottleneck data leads to improving the wrong things"),
        ("operations",
         ["Expenses reviewed", "No cost optimization discussion"],
         "Review top 3 cost centers for optimization opportunities before next quarter",
         "Proactive cost review prevents budget overruns"),
        ("operations",
         ["Vendor performance discussed", "Contract renewal dates not reviewed"],
         "Review upcoming vendor contract renewal dates and prepare negotiation positions",
         "Missing renewal windows means losing negotiation leverage"),
    ]

    for domain, kp, suggestion, reason in scenarios:
        user = build_prompt(
            None, domain, kp,
            tracked_actions=random.randint(1, 4),
            tracked_decisions=random.randint(0, 2),
            tracked_gaps=random.randint(0, 1),
            consecutive_observe=random.randint(2, 4),
            recent_agent_actions=["continue_observing"],
        )
        assistant = json.dumps({
            "action": "suggest_next_step",
            "params": {"suggestion": suggestion, "reason": reason},
            "reasoning": f"Proactive suggestion: {suggestion[:60]}",
        })
        examples.append(make_msg(user, assistant))

    return examples


# ════════════════════════════════════════════════════════════════════
# 6. provide_insight
# ════════════════════════════════════════════════════════════════════

def gen_provide_insights():
    """Generate provide_insight examples — surfacing what humans miss.

    KEY DISTINCTION from flag_gap:
    - flag_gap = something UNRESOLVED that needs a decision/owner/deadline
    - provide_insight = useful CONTEXT or WARNING the team may not be aware of

    The model should learn: if a decision WAS made but has hidden risks, or
    if the team is missing contextual knowledge — that's an insight, not a gap.
    """
    examples = []

    scenarios = [
        # ── Risk insights ──
        ("engineering", ["risk"],
         ["JWT with 24-hour expiry chosen for mobile app"],
         "24-hour JWT on mobile means users log in daily. Consider refresh tokens with secure storage for better UX without compromising security."),
        ("engineering", ["risk"],
         ["Single PostgreSQL instance chosen for 10K concurrent users"],
         "Single PostgreSQL may bottleneck at 10K concurrent users. Consider read replicas or PgBouncer from the start."),
        ("engineering", ["risk"],
         ["Using a library with only 200 GitHub stars and 1 maintainer"],
         "Low-star dependency with single maintainer is a supply chain risk. Consider a more established alternative."),
        ("engineering", ["risk"],
         ["Schema deployment without backup plan or rollback strategy"],
         "Schema changes without rollback scripts risk permanent data loss. Always create a backup first."),
        ("engineering", ["risk"],
         ["Using 5 AWS-specific services with no abstraction layer"],
         "5 AWS-specific services creates deep vendor lock-in. Consider abstraction for data services."),
        ("engineering", ["risk"],
         ["Team overtime for 3 consecutive sprints to meet deadlines"],
         "Three sprints of overtime increases burnout risk significantly. Consider reducing sprint scope."),
        ("engineering", ["risk"],
         ["Integrating with third-party API that has 99.5 percent SLA"],
         "99.5% SLA means up to 7.3 hours downtime per month. Build circuit breaker and fallback logic."),
        ("engineering", ["risk"],
         ["Auto-scaling with min 2 and max 20 instances decided"],
         "Auto-scaling has 2-5 minute cold start per new instance. Consider min 4-5 instances for sudden spikes."),
        ("engineering", ["risk"],
         ["API response payload averages 500KB for mobile clients"],
         "500KB payload is heavy for mobile. On 3G this takes 8+ seconds. Consider pagination under 50KB."),
        ("engineering", ["risk"],
         ["Single events table for all user activity data"],
         "Single events table at 100K DAU reaches 100M rows in a month. Consider partitioning by date."),
        # ── Context insights ──
        ("operations", ["context"],
         ["Switching from monthly to annual billing for 500 customers"],
         "Annual billing creates large cash inflow in month 1 but reduces monthly recurring visibility. Finance should model impact."),
        ("sales", ["context"],
         ["Preparing proposal with 10 percent discount for enterprise"],
         "Similar enterprise deals last quarter averaged 15 percent discount. Consider whether 10 percent is competitive."),
        ("operations", ["context"],
         ["Storing user data in US-only servers"],
         "US-only storage may violate GDPR for European users. Verify if any users are EU-based."),
        ("engineering", ["context"],
         ["Proposing microservices architecture for a 3-person team"],
         "Microservices typically benefit teams above 10-15 engineers. A well-structured monolith is usually better for 3 people."),
        ("sales", ["context"],
         ["Pricing product at 49 dollars per month for SMB"],
         "Industry average for similar SMB tools is 29-39 per month. Pricing at 49 requires differentiation messaging."),
        ("operations", ["context"],
         ["Hiring remote workers in 3 new states"],
         "Hiring in new states creates tax nexus obligations. HR and accounting need state tax withholding setup."),
        ("engineering", ["context"],
         ["Chose Python for a real-time low-latency trading system"],
         "Python GIL limits true parallelism. For sub-millisecond latency consider C++ or Rust for the hot path."),
        ("product", ["context"],
         ["Removing a feature used by only 5 percent of users"],
         "That 5 percent may include power users generating disproportionate revenue. Check revenue attribution first."),
        ("engineering", ["context"],
         ["3 gaps from previous sprint still unresolved but current sprint was successful"],
         "Unaddressed gaps from previous sprint tend to compound. Consider reviewing them in next planning."),
        ("sales", ["context"],
         ["Enterprise tier priced at 99 dollars per month"],
         "Competitor charges 79 for similar tier. At 99, strong differentiation messaging needed to justify premium."),
        # ── Efficiency insights ──
        ("engineering", ["efficiency"],
         ["Planning 2-week manual data migration for 50GB database"],
         "50GB migration can be automated with pg_dump/pg_restore in under 1 hour. Manual migration unnecessary."),
        ("operations", ["efficiency"],
         ["Manual regression testing every release takes 2 days"],
         "Automated regression suite could cut testing from 2 days to 30 minutes. Pays for itself after 3-4 releases."),
        ("engineering", ["efficiency"],
         ["4 CI checks running sequentially taking 45 minutes total"],
         "Running CI checks in parallel could cut pipeline time from 45 to 15 minutes."),
        ("general", ["efficiency"],
         ["Weekly 2-hour planning meeting with 12 attendees"],
         "24 person-hours per week on planning is expensive. Consider smaller focused groups and async sharing."),
        ("operations", ["efficiency"],
         ["Using 3 separate tools for project tracking, comms, and docs"],
         "Consolidating to single platform could reduce context switching and tool costs by 40 percent."),
    ]

    for domain, cats, kp, insight in scenarios:
        category = cats[0]
        user = build_prompt(
            None, domain, kp,
            tracked_actions=random.randint(0, 3),
            tracked_decisions=random.randint(0, 2),
        )
        assistant = json.dumps({
            "action": "provide_insight",
            "params": {"insight": insight, "category": category},
            "reasoning": f"Surfacing {category}: {kp[0][:50]}",
        })
        examples.append(make_msg(user, assistant))

    return examples


# ════════════════════════════════════════════════════════════════════
# 7. continue_observing
# ════════════════════════════════════════════════════════════════════

def gen_continue_observing():
    """Generate continue_observing examples — knowing when NOT to act.

    KEY: The model must learn when NOT to take action. These are situations
    where premature action would be wrong or disruptive.
    """
    examples = []

    scenarios = [
        # (domain, reason, timeline_snippet)
        # ── Meeting setup / early phase ──
        ("general", "Meeting just started, introductions happening. No actionable content yet.",
         "Host: So let us get started with introductions"),
        ("general", "Casual conversation before meeting starts. Waiting for substantive content.",
         "Someone: How was your weekend? Did you see the game?"),
        ("general", "Team reviewing agenda items. Need to wait for actual discussions.",
         "Host: OK let me go through the agenda. First we have..."),
        ("general", "Team waiting for presenter to share screen. Meeting not fully started.",
         "Presenter: Let me share my screen... hold on it is loading"),
        ("general", "Team on a break. No active discussion happening.",
         "Host: Let us take a 5 minute break. Be right back."),
        ("general", "People joining late, getting brought up to speed. No new content.",
         "Late joiner: Can someone catch me up on what I missed?"),
        # ── Routine updates without substance ──
        ("engineering", "Routine standup updates with no new decisions or tasks.",
         "Dev: All good on my end. Same priorities as yesterday."),
        ("sales", "Sales rep reading pipeline numbers from dashboard. No analysis yet.",
         "Rep: So we have 15 deals in pipeline, 5 in stage 2, 3 in stage 3"),
        ("operations", "Manager reading weekly metrics report. Data sharing, not discussing.",
         "Manager: Output was 1200 units this week, up from 1150 last week"),
        ("engineering", "Scrum master reporting sprint metrics. Standard report.",
         "SM: We completed 34 story points out of 40 planned. Pretty normal."),
        # ── Discussion in progress (premature to act) ──
        ("engineering", "Deep technical discussion still in progress. Conclusions not yet reached.",
         "Dev: What if we use WebSockets instead? Or maybe SSE? Let me think..."),
        ("product", "Two team members debating approaches. No resolution yet, let it play out.",
         "PM1: I disagree, approach B is better because... PM2: But what about cost?"),
        ("sales", "Client describing their problem at length. Need full context before acting.",
         "Client: Our current system handles 5000 orders per day and we need to scale"),
        ("operations", "Team reviewing spreadsheet data together. Analysis still in progress.",
         "Analyst: Let me scroll down to Q2 numbers... if we look at the trend"),
        ("product", "Open brainstorming session. Ideas being explored, no decisions made yet.",
         "PM: What if we tried gamification? Or a referral program? Social features?"),
        ("engineering", "Team listing out options but not yet evaluating or deciding.",
         "Lead: Options: A with Redis, B with Memcached, C with custom cache"),
        ("product", "Someone asking clarifying questions. Information gathering phase.",
         "PM: What exactly do you mean by real-time? Sub-second? Or within 5 seconds?"),
        ("engineering", "Team silently reading a shared document. No verbal content to act on.",
         "Lead: Let me give everyone 2 minutes to read through this RFC"),
        # ── Non-actionable statements ──
        ("general", "Discussion is vague and exploratory. No clear actions or decisions emerging.",
         "Someone: We should think about this more carefully. Lots of factors."),
        ("engineering", "Team sharing opinions but not converging on anything.",
         "Dev1: Architecture is fine. Dev2: It needs work. Dev3: Well it depends"),
        ("engineering", "Reviewing previous action items. No new information or assignments.",
         "PM: Last week Sarah was on OAuth. Sarah: Still in progress"),
        ("general", "Someone giving a presentation. Content being delivered, not discussed yet.",
         "Presenter: Moving to slide 5, this shows growth over the last quarter"),
        ("sales", "Just reporting numbers with no decisions or actions attached.",
         "VP: Revenue is 2.3 million, up 15 percent quarter over quarter"),
        ("engineering", "Team joking around. Not a real concern or action.",
         "Dev: Oh great, another meeting about meetings! Just kidding, on track"),
        ("general", "Acknowledging information without any decision or assignment.",
         "PM: Good point. Had not thought about that. Let me sit with it."),
        ("product", "Product manager thinking out loud, not proposing anything concrete.",
         "PM: I wonder if our users even want this. We should look at data sometime"),
        # ── Contradictions / uncertainty ──
        ("engineering", "Team members disagreeing. Wait for consensus before logging.",
         "Dev1: Use Redis. Dev2: No Memcached. Dev3: Actually lets use DynamoDB"),
        ("product", "Speaker using tentative language suggesting no firm commitment.",
         "PM: Maybe we could potentially consider doing something like a redesign"),
        ("engineering", "Team revisiting a previously decided topic. Checking if it sticks.",
         "Lead: Are we still good with PostgreSQL? Any second thoughts? No? OK"),
    ]

    for domain, reason, tl in scenarios:
        user = build_prompt(
            None, domain,
            ["Meeting content so far is non-actionable or exploratory"],
            timeline=[tl],
            consecutive_observe=0,
            tracked_actions=0, tracked_decisions=0, tracked_gaps=0,
        )
        assistant = json.dumps({
            "action": "continue_observing",
            "params": {"reason": reason},
            "reasoning": f"No action needed: {reason[:60]}",
        })
        examples.append(make_msg(user, assistant))

    return examples


# ════════════════════════════════════════════════════════════════════
# 8. Edge cases + Contrastive pairs
# ════════════════════════════════════════════════════════════════════

def gen_edge_cases():
    """Generate hard edge cases and contrastive pairs.

    Contrastive pairs explicitly teach the difference between:
    - provide_insight vs flag_gap (main confusion source)
    - suggest_next_step vs provide_insight
    - continue_observing vs log_decision (opinions vs decisions)
    """
    examples = []

    # ── Edge: Ambiguous assignment ──
    examples.append(make_msg(
        build_prompt(None, "engineering",
                     ["Discussed code quality but no specific actions were assigned"],
                     timeline=["Someone: Should probably look into that at some point"]),
        json.dumps({"action": "flag_gap", "params": {"topic": "Code quality investigation", "gap_type": "no_owner", "suggestion": "Clarify what needs improvement and assign a specific owner"}, "reasoning": "Vague reference with no specific owner or task"})
    ))

    # ── Edge: Two tasks in one sentence ──
    examples.append(make_msg(
        build_prompt(None, "engineering",
                     ["Payment system needs fix", "Docs need update"],
                     timeline=["Lead: Mike will fix the payment bug and Sarah will update docs by Friday"]),
        json.dumps({"action": "extract_action_item", "params": {"owner": "Mike", "task": "Fix payment system bug", "deadline": "Friday", "priority": "high"}, "reasoning": "Payment bug is higher priority. Sarah's task captured in next cycle."})
    ))

    # ── Edge: Contradictory statements ──
    examples.append(make_msg(
        build_prompt(None, "engineering",
                     ["Previously agreed on PostgreSQL", "Now someone suggests MongoDB"],
                     timeline=["Dev: Actually maybe MongoDB is better for our use case after all"]),
        json.dumps({"action": "continue_observing", "params": {"reason": "Prior decision being reconsidered. Wait for team to reach new consensus."}, "reasoning": "Contradicts earlier decision — should not log until resolved"})
    ))

    # ── Edge: Implicit task, no explicit assignment ──
    examples.append(make_msg(
        build_prompt(None, "engineering",
                     ["Deployment is on the roadmap", "Monitoring not set up for production"],
                     timeline=["Dev: We really need monitoring before we go to production"]),
        json.dumps({"action": "flag_gap", "params": {"topic": "Production monitoring setup", "gap_type": "no_owner", "suggestion": "Assign someone to set up monitoring before deployment"}, "reasoning": "Important task identified but not assigned to anyone"})
    ))

    # ── Edge: Meeting wrapping up ──
    examples.append(make_msg(
        build_prompt(None, "engineering",
                     ["8 key points logged", "Multiple discussions complete"],
                     timeline=["Lead: I think we have covered everything. Good meeting."],
                     tracked_actions=3, tracked_decisions=2, tracked_gaps=1),
        json.dumps({"action": "request_artifact", "params": {"artifact_type": "meeting_summary", "context_summary": "Sprint planning complete: 3 actions, 2 decisions, 1 gap.", "domain": "engineering"}, "reasoning": "Meeting ending with enough content for summary"})
    ))

    # ── Edge: Urgent P0 ──
    examples.append(make_msg(
        build_prompt(None, "engineering",
                     ["Data breach detected", "Customer PII may be exposed"],
                     timeline=["Security: This is P0. Disable the endpoint NOW and notify legal immediately."]),
        json.dumps({"action": "extract_action_item", "params": {"owner": "On-call engineer", "task": "Disable compromised endpoint and initiate incident response", "deadline": "immediate", "priority": "high"}, "reasoning": "P0 security incident requires immediate action"})
    ))

    # ── Edge: Decision with clear rationale ──
    examples.append(make_msg(
        build_prompt(None, "engineering",
                     ["Using Redis for caching", "Evaluated Memcached as alternative"],
                     timeline=["Architect: Redis supports Pub/Sub which we might need. Lets stick with Redis."]),
        json.dumps({"action": "log_decision", "params": {"decision": "Use Redis for caching over Memcached", "alternatives_rejected": ["Memcached"], "rationale": "Redis Pub/Sub support for future needs despite Memcached being faster"}, "reasoning": "Technical decision with clear rationale"})
    ))

    # ── Edge: Repeated topic → INSIGHT (gap was already flagged) ──
    examples.append(make_msg(
        build_prompt(None, "engineering",
                     ["Database migration raised 3 times today without resolution"],
                     timeline=["Lead: We keep coming back to migration. We really need to assign someone."],
                     tracked_gaps=1),
        json.dumps({"action": "provide_insight", "params": {"insight": "Database migration raised 3 times without resolution. This recurring pattern suggests it needs escalation.", "category": "efficiency"}, "reasoning": "Recurring topic — insight not gap because gap was already flagged"})
    ))

    # ── Edge: Long meeting ending ──
    examples.append(make_msg(
        build_prompt(None, "engineering",
                     ["15+ points discussed", "Meeting running 90 minutes"],
                     timeline=["Lead: OK we have been going a while. Any last things?"],
                     tracked_actions=5, tracked_decisions=4, tracked_gaps=2,
                     consecutive_observe=4),
        json.dumps({"action": "request_artifact", "params": {"artifact_type": "meeting_summary", "context_summary": "Extended planning: 5 actions, 4 decisions, 2 gaps. Summary for absent members.", "domain": "engineering"}, "reasoning": "Long meeting ending — comprehensive summary needed"})
    ))

    # ── Edge: Task with no deadline ──
    examples.append(make_msg(
        build_prompt(None, "product",
                     ["Privacy policy needs review and update"],
                     timeline=["PM: Nina, can you look at the privacy policy when you get a chance?"]),
        json.dumps({"action": "extract_action_item", "params": {"owner": "Nina", "task": "Review and update privacy policy", "deadline": "unspecified", "priority": "low"}, "reasoning": "Task assigned but 'when you get a chance' implies low priority"})
    ))

    # ── Edge: Multiple teams, parallel tracks ──
    examples.append(make_msg(
        build_prompt(None, "engineering",
                     ["Frontend team discussing UX redesign", "Backend team discussing API v2"],
                     timeline=["Director: Frontend handles UX, backend starts API v2. Both ready by month end."]),
        json.dumps({"action": "extract_action_item", "params": {"owner": "Backend team", "task": "Start API v2 migration", "deadline": "end of month", "priority": "high"}, "reasoning": "API v2 is the higher-risk track to prioritize"})
    ))

    # ── Edge: Decision reversal ──
    examples.append(make_msg(
        build_prompt(None, "engineering",
                     ["Previously decided PostgreSQL", "New benchmarks show scaling issues"],
                     timeline=["CTO: Given new benchmarks we are reversing. Go with CockroachDB."]),
        json.dumps({"action": "log_decision", "params": {"decision": "Switch from PostgreSQL to CockroachDB", "alternatives_rejected": ["PostgreSQL (previous choice)"], "rationale": "New benchmarks show PostgreSQL scaling issues"}, "reasoning": "Decision reversed based on new data"})
    ))

    # ══════════════════════════════════════════════════════════════
    # CONTRASTIVE PAIRS: Teaching insight vs gap vs suggestion
    # ══════════════════════════════════════════════════════════════

    # CONTRASTIVE: Decision was made → INSIGHT (context), NOT gap
    examples.append(make_msg(
        build_prompt(None, "sales",
                     ["Enterprise tier priced at 99 dollars per month", "Decision is final"],
                     timeline=["VP: OK 99 dollars for enterprise, that is final"],
                     tracked_decisions=1),
        json.dumps({"action": "provide_insight", "params": {"insight": "Competitor charges 79 for similar tier. At 99 you need strong differentiation messaging.", "category": "context"}, "reasoning": "Decision made (not a gap), but team may not know competitor pricing"})
    ))

    # CONTRASTIVE: No decision reached → GAP, NOT insight
    examples.append(make_msg(
        build_prompt(None, "sales",
                     ["Enterprise pricing discussed but no agreement on amount"],
                     timeline=["VP: We need to figure out pricing but lets move on for now"]),
        json.dumps({"action": "flag_gap", "params": {"topic": "Enterprise pricing decision", "gap_type": "no_decision", "suggestion": "Prepare pricing proposals with competitive analysis for next meeting"}, "reasoning": "Pricing discussed without decision — this is a gap"})
    ))

    # CONTRASTIVE: Decision made with hidden risk → INSIGHT
    examples.append(make_msg(
        build_prompt(None, "engineering",
                     ["Decided to store all events in single table", "Decision logged"],
                     timeline=["Lead: One events table keeps things simple"],
                     tracked_decisions=2),
        json.dumps({"action": "provide_insight", "params": {"insight": "Single events table grows fast. At 100K DAU this reaches 100M rows/month. Partition by date from the start.", "category": "risk"}, "reasoning": "Decision was already made, but there is a scaling risk the team may not see"})
    ))

    # CONTRASTIVE: Missing discussion topic → SUGGESTION (agenda gap)
    examples.append(make_msg(
        build_prompt(None, "product",
                     ["Feature list and marketing plan discussed", "No mention of support scaling"],
                     timeline=["PM: I think we have a solid launch plan"],
                     consecutive_observe=3, tracked_actions=2, tracked_decisions=1),
        json.dumps({"action": "suggest_next_step", "params": {"suggestion": "Discuss customer support scaling plan for launch", "reason": "Launch plan has no support readiness. Post-launch failures damage brand."}, "reasoning": "Team overlooked critical launch component — suggesting they discuss it"})
    ))

    # CONTRASTIVE: Opinion without consensus → CONTINUE_OBSERVING
    examples.append(make_msg(
        build_prompt(None, "engineering",
                     ["Discussing whether to add TypeScript to the codebase"],
                     timeline=["Dev: I think TypeScript would be helpful but I could go either way"]),
        json.dumps({"action": "continue_observing", "params": {"reason": "Personal opinion expressed, no decision made. Other team members have not weighed in."}, "reasoning": "One person expressing preference is not a decision — wait for consensus"})
    ))

    # CONTRASTIVE: Celebrating but have unresolved issues → INSIGHT
    examples.append(make_msg(
        build_prompt(None, "engineering",
                     ["Sprint completed on time", "Team celebrating"],
                     timeline=["PM: Great sprint everyone! Everything went smoothly."],
                     tracked_gaps=3),
        json.dumps({"action": "provide_insight", "params": {"insight": "3 gaps from previous sprint remain unresolved. Unaddressed gaps compound. Review them in next planning.", "category": "context"}, "reasoning": "Team celebrating but has unresolved issues from before"})
    ))

    # CONTRASTIVE: Technical knowledge addition → INSIGHT (not suggestion)
    examples.append(make_msg(
        build_prompt(None, "engineering",
                     ["Auto-scaling config decided: min 2, max 20 instances"],
                     timeline=["Ops: Scale from 2 to 20 should handle anything"]),
        json.dumps({"action": "provide_insight", "params": {"insight": "Auto-scaling has 2-5 minute cold start per instance. Consider min 4-5 to absorb initial traffic surge.", "category": "risk"}, "reasoning": "Team may not be aware of cold start implications"})
    ))

    return examples


# ════════════════════════════════════════════════════════════════════
# Main: generate, shuffle, split, write
# ════════════════════════════════════════════════════════════════════

def main():
    all_examples = []

    # Generate all categories
    all_examples.extend(gen_extract_action_items())
    all_examples.extend(gen_log_decisions())
    all_examples.extend(gen_flag_gaps())
    all_examples.extend(gen_request_artifacts())
    all_examples.extend(gen_suggest_next_steps())
    all_examples.extend(gen_provide_insights())
    all_examples.extend(gen_continue_observing())
    all_examples.extend(gen_edge_cases())

    # Shuffle for training
    random.shuffle(all_examples)

    # Write to file
    with open(OUTPUT, "w") as f:
        for example in all_examples:
            f.write(json.dumps(example) + "\n")

    # Print distribution
    dist = {}
    for ex in all_examples:
        action = json.loads(ex["messages"][2]["content"])["action"]
        dist[action] = dist.get(action, 0) + 1

    print(f"\n✅ Generated {len(all_examples)} training examples → {OUTPUT}")
    print(f"\nDistribution:")
    for action, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {action}: {count} ({count/len(all_examples)*100:.0f}%)")


if __name__ == "__main__":
    main()
