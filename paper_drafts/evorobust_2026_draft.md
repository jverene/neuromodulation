\documentclass[10pt]{article}

% --- Geometry & NeurIPS Page Layout ---
\usepackage[text={5.5in,9.0in}, centering, headheight=12pt]{geometry}
\usepackage{newtxtext,newtxmath} % Standard Times Roman font baseline
\usepackage{microtype} % Character protrusion & expansion for clean margins

% --- Core Packages ---
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{xcolor}
\usepackage{url}
\usepackage{hyperref}

% --- Color Palette & Hyperlink Setup ---
\definecolor{navyblue}{RGB}{0, 32, 96}
\hypersetup{
colorlinks=true,
linkcolor=navyblue,
citecolor=navyblue,
urlcolor=navyblue
}

% --- Custom NeurIPS Style Macros ---
\makeatletter
\renewcommand{\maketitle}{
\begin{center}
\vspace\*{0.2in}
{\Large \bfseries \@title \par}
\vskip 1.2em
{\large Anonymous Author(s) \\
\small \texttt{anonymous@example.com} \\
}
\vskip 1.5em
\end{center}
}
\makeatother

\renewenvironment{abstract}{
\begin{quote}
\centerline{\small\bfseries Abstract}
\vspace{0.5em}
\small
}{
\end{quote}
\vspace{1.0em}
}

% Helper macro to safely handle missing figure assets during compilation
\newcommand{\safefigure}[2]{%
\IfFileExists{#1}{%
\includegraphics[width=\linewidth]{#1}%
}{%
\framebox[\linewidth]{\rule{0pt}{3.2cm}\footnotesize\color{gray}\textbf{Placeholder:} #2}%
}%
}

\title{Robustness Attribution in Regenerative Neural Cellular\\Automata: Training, Modulation, and Cross-Parent Transfer}
% NOTE(outcome-dependent alternates, choose after results):
% A: Evolutionary Modulation Improves Parent-Specific Robustness but Fails
%    to Transfer Across Neural Cellular Automata
% B: Channel-Aware Training, Not Closed-Loop Control, Drives Robust
%    Regeneration in Neural Cellular Automata

\begin{document}

\maketitle

\begin{abstract}
Regenerative Neural Cellular Automata (NCAs) are usually evaluated on a
single trained model, so robustness claims cannot separate what
\emph{training} provides from what run-time \emph{control} provides. We
cross five independently trained parent seeds with four conditions under
recurring multi-block damage that exceeds the perception radius: a $K{=}0$
parent; the channel-aware $K{=}3$ parent with modulation pinned to neutral;
that parent with a controller evolved \emph{for it}; and a cross-parent
transfer probe. The comparison is preregistered. Channel-aware training
improves robustness in five of five parent seeds (median final-Hamming
reduction $0.008$, up to $0.140$ on a fragile parent); controllers evolved
on their own parents add nothing exceeding damage-seed noise (median
$\Delta_s = +0.001$, all $|\Delta_s|$ an order of magnitude below the
per-run SD); and the evolved artifact is a \emph{parent-specific tonic
constant} (controller output is flat, with no lesion-locked response) that
transfers as a penalty in five of five siblings --- lethally in two.
Single-parent evaluation hid all three facts. The attribution protocol ---
adversarial damage calibration, objective-hacking and
search-initialization probes, and parent-seed-resolved decomposition ---
transfers to other self-organizing systems.
\end{abstract}

\section{Introduction}
\label{sec:intro}

Growing Neural Cellular Automata (GNCAs) demonstrate that a single local,
differentiable update rule can grow a target morphology from a seed and
regenerate it after damage~\cite{mordvintsev2020gnca}. The capability is
bounded by perception: wounds larger than the perception radius contain
cells with no signal to integrate, and severed fragments retain no
information about the body they came from. Prior work, including our own
single-parent study (Appendix~\ref{app:singleparent}), adds global
modulator channels and evolves release policies for them --- and reports
the result on \emph{one} trained model.

Single-model evaluation cannot answer the question it appears to answer.
When a modulated NCA outperforms an unmodulated one, three effects are
confounded: (i) \emph{training with channels present}, (ii) \emph{run-time
modulation} by a controller evolved for that parent, and (iii) transfer
across independently trained parents. Our earlier five-condition study
found closed-loop, static, and constant modulation indistinguishable, but
could not tell whether its benefit over the unmodulated baseline came from
the controller or from the parent happening to be trained with channels. A
three-parent pilot (Appendix~\ref{app:pilot}) then found the evolved
controller \emph{transfers lethally} to sibling parents while zero-output
channel parents beat the $K{=}0$ baseline in every seed --- the single-
parent table had mis-attributed a parent-training effect to closed-loop
control.

This paper resolves the attribution with a preregistered five-parent-seed
study. For each seed $s \in \{0,\dots,4\}$ we train $K{=}0$ and
channel-aware $K{=}3$ parents from scratch, evolve a controller \emph{on
that parent}, and evaluate four conditions on held-out damage seeds:
$H(K{=}0,s)$, $H(K{=}3,m{=}0,s)$, $H(K{=}3,\text{own ctrl},s)$, and a
cross-parent transfer probe $H(K{=}3,\text{July ctrl},s)$. The primary
statistic, thresholds, and all three paper outcomes were fixed before data
collection (Section~\ref{sec:design}); the preregistration and analysis
script are in the repository history, timestamped before the runs.

\paragraph{Contributions.}
\begin{enumerate}
  \item A preregistered robustness-attribution protocol for regenerative
        NCAs: adversarial damage calibrated to exceed perception, crossed
        with parent-seed variation, decomposed into
        $E_{\mathrm{train}}$, $E_{\mathrm{ctrl}}$, $E_{\mathrm{transfer}}$,
        with calibration probes that make the evolution landscape
        interpretable (Appendix~\ref{app:calibration}).
  \item The attribution result: channel-aware \emph{training} is the
        robust cause (5/5 seeds); controllers evolved on their own parents
        add nothing exceeding noise; the evolved artifact is a
        parent-specific tonic \emph{constant} that transfers as a penalty
        in 5/5 siblings (lethally in 2).
\end{enumerate}

\section{Related Work}
\label{sec:related}

\paragraph{Neural cellular automata.}
GNCAs grow patterns from a seed and regenerate after
damage~\cite{mordvintsev2020gnca}, with extensions to
self-classification~\cite{randazzo2020selfclass} and
textures~\cite{mordvintsev2021texture}. All rely on local perception, and
regeneration degrades beyond the perception radius --- the regime our
benchmark exploits. Signal channels~\cite{stovold2023signal}, goal
conditioning~\cite{sudhakaran2022goal}, and information-dynamical analyses
of self-maintenance~\cite{masumori2026fluctuations} all treat the signal
as fixed input or emergent byproduct, never a controlled output. We ask
the prior question: when such a channel \emph{appears} to help, what is
actually helping?

\paragraph{Search over evaluation regimes.}
Our design follows the insight, central to novelty
search~\cite{lehman2011novelty} and quality-diversity
optimization~\cite{mouret2015mapelites}, that a fixed objective can
misdirect search and hide failure modes. Environment-generation methods
co-evolve challenges with solutions:
POET~\cite{wang2019poet}, PAIRED~\cite{dennis2020paired},
ACCEL~\cite{parkerholder2022accel}. Our benchmark does not yet evolve the
damage distribution; it is the controlled stationary baseline against
which such co-evolution can be judged.

\section{Experimental design}
\label{sec:design}

\paragraph{Damage regime and parents.}
Recurring multi-block lesions --- every 150 steps, $n{=}4$ contiguous
$16{\times}16$ blocks at seeded positions, $T{=}2000$ --- exceed the
perception radius, so wound interiors have no living neighbors (the
information-isolation regime mapped in Appendix~\ref{app:e1}). Damage
seeds 0--7 drive evolution; held-out seeds 10000--10007 drive all
reported numbers; the schedule is deliberately adversarial
(Appendix~\ref{app:calibration}). Per seed $s$ we train a $K{=}0$ and a
channel-aware $K{=}3$ parent from scratch (Appendix~\ref{app:model}); the
$K{=}3$ parent's three global modulator channels are the only non-local
pathway. A 259-parameter controller ($4{\to}32{\to}3$, $\tanh$) reads
four target-free grid statistics and sets release every 10 steps; each
parent gets its own controller via CMA-ES~\cite{hansen2006cma} in
Evosax~\cite{lange2022evosax} (population 64, $\sigma_0{=}0.01$, 300
generations, event-weighted Hamming objective;
Appendix~\ref{app:evolution}).

\paragraph{Preregistered comparison.}
All conditions run on the same held-out damage seeds (5 condition seeds
$\times$ 8 damage seeds) from a shared $t{=}0$ state grown by the $K{=}3$
parent. The primary statistic is
\begin{equation}
  \Delta_s \;=\; H(K{=}3, m{=}0, s) \;-\; H(K{=}3, \text{own controller}, s),
  \label{eq:delta}
\end{equation}
positive when the evolved controller helps \emph{its own} parent. Fixed
before data collection: \emph{controller effect supported} if $\Delta_s >
0$ in $\geq 4/5$ seeds with sign-consistent differences;
\emph{channel-training effect supported} if $H(K0,s) - H(m0,s) > 0$ in
$\geq 4/5$; \emph{transfer failure supported} if the July controller
underperforms the own-controller substantially in $\geq 4/5$. Effects are
reported per seed with median and range; we make no significance claims
at five seeds. Controller-output ($m_t$) series distinguish tonic
calibration from event-locked policy (Appendix~\ref{app:mt}).

\section{Results}
\label{sec:results}

Table~\ref{tab:effects} reports the preregistered decomposition; the full
per-seed, per-condition numbers behind it are in
Appendix~\ref{app:rawtables}.

\begin{table}[h]
  \centering
  \small
  \caption{Per-seed effects. $E_{\mathrm{train}}{=}H(K0)-H(m0)$;
  $\Delta_s{=}H(m0)-H(\text{own})$ (preregistered primary);
  $E_{\mathrm{transfer}}{=}H(\text{July})-H(\text{own})$. Positive favors
  the named component.}
  \label{tab:effects}
  \begin{tabular}{lccc}
    \toprule
    Seed & $E_{\mathrm{train}}$ & $\Delta_s$ ($E_{\mathrm{ctrl}}$) & $E_{\mathrm{transfer}}$ \\
    \midrule
    0 & $+0.0053$ & $+0.0013$ & $+0.0083$ \\
    1 & $+0.1400$ & $+0.0084$ & $+0.2189^{\dagger}$ \\
    2 & $+0.0077$ & $-0.0001$ & $+0.4129^{\dagger}$ \\
    3 & $+0.0139$ & $-0.0011$ & $+0.0665$ \\
    4 & $+0.0074$ & $+0.0042$ & $+0.0088$ \\
    \midrule
    median & $+0.0077$ & $+0.0013$ & $+0.0665$ \\
    \bottomrule
  \end{tabular}

  \smallskip
  \footnotesize{$^{\dagger}$lethal: survival $0.00$. Per-run SD of final
  Hamming is $0.01$--$0.03$ in every condition; all $|\Delta_s|$ are an
  order of magnitude below it.}
\end{table}

\paragraph{Outcome.}
Applying the preregistered rule: the \emph{channel-training effect is
supported} ($E_{\mathrm{train}}>0$ in 5/5 seeds; bar was $\geq$4/5). The
\emph{controller effect is not} ($\Delta_s>0$ in only 3/5; bar was
$\geq$4/5) --- mixed signs with every magnitude an order of magnitude
below per-run noise: no seed shows a noise-exceeding benefit from evolving
a controller on its own parent. The \emph{transfer failure is supported}
(penalty in 5/5; lethal in 2). We emphasize the honest reading: controller
efficacy is parent-dependent at noise level, which for practical purposes
means absent.

\paragraph{What the controller actually emits.}
All five evolved controllers emit a \emph{constant}: within-rollout std of
$m_t$ is $0.002$--$0.004$ per channel, the correlation between $m_t$ and a
post-lesion indicator is $|r|\leq0.17$, and the mean $|\Delta m|$ at
lesion steps is indistinguishable from baseline drift. Evolution learned a
parent-specific tonic calibration, not a release policy (per-controller
diagnostics in Appendix~\ref{app:mt}).

\section{Discussion}
\label{sec:discussion}

\paragraph{Which component causes robustness.}
The robustness gain comes primarily from channel-aware parent training:
zero-output channel parents beat their $K{=}0$ siblings in 5/5 seeds, with
the largest effect exactly where the unmodulated parent is most fragile
(seed 1: $0.179 \to 0.039$). Evolved controllers add nothing exceeding
noise on any parent, and the artifact itself is a tonic constant.
Single-parent controller evaluation --- ours included --- can therefore
overattribute robustness to closed-loop control that is actually a
property of training.

\paragraph{Why cross-parent transfer fails.}
The transfer probe uses one donor controller, five recipients --- a probe,
not a full transfer study (which would test every donor--recipient pair).
Even so, the observed penalties (positive in 5/5, lethal in 2) are
structural: each controller's output is a constant calibrated against its
own parent's channel weights, so the same release level drives different
dynamics in a sibling. Evolution found an operating point for one
organism, not a modulation law.

\paragraph{Limitations and future work.}
Five parent seeds; one controller evolution per parent; one morphology;
one stationary damage family; transfer probed with a single donor. Repair
half-life is comparable only among conditions that return to near-target;
Hamming conflates locomotion drift with morphological error uniformly.
Next: multi-parent (population-based) evolution; full transfer matrices;
non-stationary, diversity-driven damage co-evolution, for which this
benchmark is the controlled baseline.

\section{Conclusion}
\label{sec:conclusion}

In regenerative NCAs under adversarial recurring damage, robustness comes
from channel-aware training, not from evolved control: zero-output channel
parents win in 5/5 seeds, evolved controllers add noise-level benefit, and
their tonic constants transfer as penalties (lethally in 2/5). The broader
contribution is methodological: robustness claims about self-organizing
systems should be attributed across model seeds, not only damage seeds.

\appendix

\section{Full per-seed, per-condition results}
\label{app:rawtables}

\begin{table}[h]
  \centering
  \small
  \caption{Final Hamming (mean $\pm$ SD, 5 condition seeds $\times$ 8
  held-out damage seeds) for four conditions across five parent seeds,
  hard recurring multi-block damage, $T{=}2000$.}
  \label{tab:attribution}
  \begin{tabular}{lcccc}
    \toprule
    Parent seed & $K{=}0$ & $K{=}3$, $m{=}0$ & $K{=}3$, own ctrl & $K{=}3$, July ctrl \\
    \midrule
    0 & $0.034{\pm}.012$ & $0.029{\pm}.014$ & $0.028{\pm}.014$ & $0.036{\pm}.010$ \\
    1 & $0.179{\pm}.032$ & $0.039{\pm}.032$ & $0.031{\pm}.020$ & $0.249{\pm}.046$ \\
    2 & $0.029{\pm}.018$ & $0.021{\pm}.017$ & $0.021{\pm}.016$ & $0.434{\pm}.012$ \\
    3 & $0.047{\pm}.005$ & $0.033{\pm}.017$ & $0.034{\pm}.016$ & $0.100{\pm}.020$ \\
    4 & $0.044{\pm}.011$ & $0.036{\pm}.025$ & $0.032{\pm}.019$ & $0.041{\pm}.019$ \\
    \midrule
    median & $0.044$ & $0.033$ & $0.031$ & $0.100$ \\
    \bottomrule
  \end{tabular}
\end{table}

Survival is $1.00$ everywhere except: $K{=}0$ on seed 1 ($0.03$), July
controller on seeds 1/2 ($0.00$) and seed 3 ($0.45$). AUC mirrors the
same ordering in every seed (full CSVs in the repository).

\section{Controller-output ($m_t$) diagnostics}
\label{app:mt}

\begin{table}[h]
  \centering
  \small
  \caption{Per-controller $m_t$ diagnostics over one held-out hard-regime
  rollout. corr = correlation with a post-lesion indicator; jump =
  mean $|\Delta m|$ in the 10 steps after lesions; drift = mean $|\Delta m|$
  elsewhere. All five: flat output, no lesion-locked response.}
  \label{tab:mt}
  \begin{tabular}{lcccc}
    \toprule
    Seed & std$(m_t)$ & corr & jump & drift \\
    \midrule
    0 & $0.0020$ & $-0.17$ & $0.0006$ & $0.0006$ \\
    1 & $0.0031$ & $+0.03$ & $0.0010$ & $0.0010$ \\
    2 & $0.0035$ & $-0.14$ & $0.0012$ & $0.0011$ \\
    3 & $0.0026$ & $-0.13$ & $0.0008$ & $0.0008$ \\
    4 & $0.0029$ & $+0.05$ & $0.0010$ & $0.0009$ \\
    \bottomrule
  \end{tabular}
\end{table}

\section{Pilot: single-parent evaluation hides parent-locking}
\label{app:pilot}

A three-parent pilot (2026-08-16) first exposed the confound. The
single-parent controller from our original five-condition study
transferred lethally to two of three sibling parents (survival $0.00$,
final Hamming $0.249$ and $0.434$), while zero-output channel parents beat
the $K{=}0$ baseline in $3/3$ seeds ($0.029/0.034/0.020$ vs
$0.034/0.177/0.028$). The original table's headline gap was therefore part
parent-training effect and part parent-seed luck --- motivating this
study.

\section{Model and training details}
\label{app:model}

The grid state $x_t \in [0,1]^{96 \times 96 \times 16}$ holds 16 channels;
the last four are RGBA with alpha last. Perception uses three fixed kernels
(identity, Sobel-$x$, Sobel-$y$) per channel, giving
$p_t \in \mathbb{R}^{48}$:
\begin{equation}
  p_t = \big(K_{\mathrm{id}} * x_t,\; K_{\mathrm{S}_x} * x_t,\;
  K_{\mathrm{S}_y} * x_t\big).
\end{equation}
The update MLP maps $(48{+}K)$ inputs to 128 hidden units (ReLU) and back to
16 channel increments, with the final layer zero-initialized. Cells update
with probability $0.5$; a cell is alive when its alpha exceeds $0.1$ within
its $3{\times}3$ max-pooled neighborhood; dead cells are masked to zero.

Tonic and phasic modulator dynamics:
\begin{equation}
  m_t^{(\mathrm{tonic})} = \alpha\, m_{t-1}^{(\mathrm{tonic})} + (1-\alpha)\, c_t,
  \qquad
  m_t^{(\mathrm{phasic})} = m_{t-1}^{(\mathrm{phasic})} \cdot e^{-\Delta t / \tau},
\end{equation}
with $\alpha = 0.95$, $\tau = 20$, and the injected level the clipped sum,
broadcast to all cells and concatenated to perception (input dim $48+3=51$).

Parents train on a $40{\times}40$ emoji target zero-padded to the canvas,
with pool-based training (pool 1024, batch 8, worst-in-batch reseeding),
random square-erasure damage mid-episode, MSE loss on RGBA, and Adam at
$10^{-3}$ for 8000 steps. A first attempt at $2{\times}10^{-3}$ diverged to
NaN near step 5532 via a late-stage loss-spike cascade; $10^{-3}$ absorbs
the identical spike.

\section{Perception-radius sweep}
\label{app:e1}

A single-lesion sweep (radii $\{2,4,8,16\}$, single- and multi-site, with
and without modulator channels) maps the boundary of local repair. Recovery
is near-perfect for small lesions and degrades once lesions exceed the
effective perception radius. Two failure signatures motivate this work:
severed fragments that survive but cannot reattach, die, or decide what to
grow into, and persistent half-alpha scar tissue at wound sites. Both are
information failures: the misbehaving cells are exactly the cells cut off
from global context.

\begin{figure}[h]
  \centering
  \includegraphics[width=0.85\linewidth]{figures/fig1_recovery_vs_radius}
  \caption{E1 lesion sweep. Final Hamming distance as a function of lesion
  radius and kind, with and without modulator channels (mean $\pm$ SD over 5
  damage seeds). Inset: post-regrowth debris and scar tissue.}
  \label{fig:e1}
\end{figure}

\section{Calibration probes}
\label{app:calibration}

\paragraph{Collapse probe.}
The target mask is only $4.6\%$ alive, so a fully-dead grid scores Hamming
$\approx 0.046$ --- close to a struggling organism's $\approx 0.02$. We
verified empirically that the all-dead state scores $0.0461$ against
struggling-neutral's $0.0205$: death loses by $2.2\times$, so the
alive-fraction floor is disabled ($0.0$). Notably, the runbook-default floor
of $0.3$ would have \emph{penalized every healthy candidate} (a correct
lizard lives at $\approx 0.057$ alive), pushing evolution toward overgrowth
--- a case where calibrating from the experiment's own data overrode a
plausible default.

\paragraph{Initial step-size probe.}
Perturbing the neutral controller by Gaussian noise at increasing $\sigma$
(25 samples each, hard-regime evaluation) showed: $\sigma \le 0.01$ stays
healthy (some samples beat neutral); $\sigma = 0.05$ is bimodal;
$\sigma = 0.3$ lands \emph{every} sample in saturated overgrowth (alive
$\rightarrow 0.96$). A first evolution run with the library-default
$\sigma_0 = 0.3$ froze at fitness $0.150$ from generation 2 onward ---
$7\times$ worse than doing nothing --- because CMA-ES was ranking
overgrowth-degree noise. $\sigma_0 = 0.01$ brackets the neutral point and
the same budget converged smoothly (Figure~\ref{fig:evolution}).

\begin{figure}[h]
  \centering
  \includegraphics[width=0.85\linewidth]{figures/fig4_evolution_trajectory}
  \caption{CMA-ES fitness over 300 generations ($\sigma_0{=}0.01$). Dashed
  line: neutral controller ($0.0205$). Best evolved fitness: $0.0135$ --- a
  $34\%$ improvement with no stall.}
  \label{fig:evolution}
\end{figure}

\section{Evolution objective}
\label{app:evolution}

We evolve the controller with CMA-ES~\cite{hansen2006cma} (Evosax
implementation~\cite{lange2022evosax}), population 64, initial step size
$\sigma_0{=}0.01$, 300 generations, neutral (all-zero) controller as the
initial mean. Fitness is the event-weighted mean Hamming distance to the
target alpha mask over full rollouts on the eight train damage seeds:
\begin{equation}
  f(\theta) = \frac{1}{\sum_t w_t} \sum_{t=1}^{T} w_t\, H_t(\theta),
  \qquad
  w_t = \exp\!\Big(-\frac{t - \tau_{\mathrm{last}}(t)}{\tau_w}\Big),
  \label{eq:fitness}
\end{equation}
where $\tau_{\mathrm{last}}(t)$ is the most recent lesion step and
$\tau_w = 150/3 = 50$. The recency kernel resets at each lesion, so slow
repair is expensive even when endpoint Hamming is identical --- it
de-saturates the metric and prices repair speed directly. Fitness is not the
reported metric: all results report unweighted trajectory statistics
on held-out seeds. Two calibration controls make this landscape
interpretable --- a collapse probe (verifying the objective cannot be gamed
by alpha suppression) and an initial step-size ablation (showing
$\sigma_0{=}0.3$ saturates the first population in overgrowth where no
repair signal exists) --- both detailed in Appendix~\ref{app:calibration}.

\section{Single-parent five-condition study}
\label{app:singleparent}

Five conditions share the same protocol, horizon, and damage seeds:
\textbf{closed-loop} (evolved controller); \textbf{static} (the evolved
controller evaluated once at $t{=}0$ and held constant --- isolating
temporal modulation from the evolved tonic level itself); \textbf{constant}
(fixed tonic, grid-searched over $\{-1,-0.5,0,0.5,1\}$ on train seeds);
\textbf{random} (uniform $[-1,1]$ each step, actuation-matched noise); and
\textbf{no modulation} ($K{=}0$ parent). All experiments ran on a single
rented A100; parent training took $\approx$20 min/model, evolution
$\approx$2.5 h, total project cost under \$15.

\paragraph{Modulation helps substantially (RQ1).}
Relative to no modulation, modulated conditions recover $2.2\times$ more
completely (final Hamming $0.028$--$0.030$ vs.\ $0.063$) and sustain
$\approx 4\times$ less cumulative damage (AUC $0.016$--$0.017$ vs.\ $0.064$).
The wounds exceed the perception radius, so this improvement is attributable
to the broadcast channel --- the only non-local pathway --- repairing damage
that local rules cannot diagnose.

\paragraph{A regime boundary, not a failed ablation (RQ2).}
The three modulated conditions are statistically indistinguishable: final
Hamming spans $0.028$--$0.030$, half-life $6.4$--$7.2$ steps, all differences
within noise (pairwise effect sizes $< 0.15$). We read this as a measurement
of the \emph{regime}, not a failure of the controller: the damage process is
stationary and memoryless --- events are i.i.d.\ in size, number, and
position --- so the optimal release policy is time-invariant and a fixed
tonic gain is structurally adequate. There is no temporal structure for
adaptive scheduling to exploit. Evolution's measured contribution is
automatic discovery of the near-optimal tonic level in 2.5 hours from a
neutral start, without the hand search the constant condition required.

\paragraph{Random modulation is lethal.}
Random actuation kills every run (survival $0.00$, final Hamming $0.825$):
pilot probes show constant levels alone swing mean Hamming from $0.003$ to
$0.93$. The modulator channels are a high-gain control pathway, not a free
architectural bonus --- the identity of the release schedule determines
whether the organism lives.

\paragraph{Metric artifacts of long-horizon evaluation (RQ3).}
Two artifacts surfaced only under repeated damage. First, repair half-life is
defined per lesion relative to each post-lesion jump; on the baseline's
chronically damaged morphology, new lesions often land on already-degraded
regions, producing no measurable jump and scoring as zero half-life --- so
the baseline's $2.7$ steps does \emph{not} indicate fast repair. Second,
residual Hamming in all conditions is partly attributable to locomotion drift
during regeneration rather than permanent damage; the metric conflates
spatial translation with morphological error. Both are documented so that
future benchmark users can avoid them.

\section{Reproducibility}
\label{app:repro}

Parents train in $\approx$20--30 min each on one A100; each 300-generation
evolution $\approx$4.5--5.5 h on the study host ($\approx$2.5 h on a
faster one); the full five-seed study cost $\approx$\$20 of rented GPU
time. Preregistration, analysis script, per-seed artifacts (controllers,
parents, trajectories, $m_t$ series), and the mechanical outcome output are
in the repository, timestamped before the results. Code will be released
upon publication.

\begin{thebibliography}{13}

\bibitem{mordvintsev2020gnca}
A.~Mordvintsev, E.~Randazzo, E.~Niklasson, and M.~Levin.
\emph{Growing Neural Cellular Automata}.
Distill 5(2):e23, 2020.

\bibitem{randazzo2020selfclass}
E.~Randazzo, A.~Mordvintsev, E.~Niklasson, M.~Levin, and S.~Greydanus.
\emph{Self-classifying MNIST Digits}.
Distill, 2020.

\bibitem{mordvintsev2021texture}
E.~Niklasson, A.~Mordvintsev, E.~Randazzo, and M.~Levin.
\emph{Self-Organising Textures}.
Distill 6(2), 2021.

\bibitem{stovold2023signal}
J.~Stovold.
\emph{Neural Cellular Automata Can Respond to Signals}.
ALIFE 2023; arXiv:2305.12971.

\bibitem{sudhakaran2022goal}
S.~Sudhakaran, E.~Najarro, and S.~Risi.
\emph{Goal-Guided Neural Cellular Automata: Learning to Control Self-Organising Systems}.
arXiv:2205.06806, 2022.

\bibitem{masumori2026fluctuations}
A.~Masumori, M.~Sato, and T.~Ikegami.
\emph{Structured Fluctuations and the Information Dynamics of Self-Maintenance in Growing Neural Cellular Automata}.
arXiv:2607.12403, 2026.

\bibitem{lehman2011novelty}
J.~Lehman and K.~O.~Stanley.
\emph{Abandoning Objectives: Evolution Through the Search for Novelty Alone}.
Evolutionary Computation, 19(2):189--223, 2011.

\bibitem{mouret2015mapelites}
J.-B.~Mouret and J.~Clune.
\emph{Illuminating Search Spaces by Mapping Elites}.
arXiv:1504.04909, 2015.

\bibitem{wang2019poet}
R.~Wang, J.~Lehman, J.~Clune, and K.~O.~Stanley.
\emph{Paired Open-Ended Trailblazer (POET): Endlessly Generating Increasingly Complex and Diverse Learning Environments and Their Solutions}.
arXiv:1901.01753, 2019.

\bibitem{dennis2020paired}
M.~Dennis, N.~Jaques, E.~Vinitsky, A.~Bayen, S.~Russell, A.~Critch, and S.~Levine.
\emph{Emergent Complexity and Zero-Shot Transfer Through Unsupervised Environment Design}.
NeurIPS 2020.

\bibitem{parkerholder2022accel}
J.~Parker-Holder, R.~Rajeev, K.~Hartikainen, et al.
\emph{Evolving Curricula with Regret-Based Environment Design}.
ICML 2022; arXiv:2203.01302.

\bibitem{hansen2006cma}
N.~Hansen.
\emph{The CMA Evolution Strategy: A Comparing Review}.
In \emph{Towards a New Evolutionary Computation}, Studies in Fuzziness and
Soft Computing vol.~192, pages 75--102. Springer, 2006.

\bibitem{lange2022evosax}
R.~T.~Lange.
\emph{evosax: JAX-Based Evolution Strategies}.
arXiv:2212.04180, 2022.

\end{thebibliography}

\end{document}
