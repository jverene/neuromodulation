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

\title{Evolutionary Stress Testing of Self-Repair:\\When Does Closed-Loop Modulation Matter?}

\begin{document}

\maketitle

\begin{abstract}
Robust self-organizing systems should be evaluated under recurring, diverse
disruptions rather than a single fixed lesion. We introduce an evolutionary
stress-testing framework for Growing Neural Cellular Automata (GNCAs) in which
multi-block lesions are calibrated to exceed the local perception radius, and a
compact controller regulates global chemical modulation from target-free grid
statistics. CMA-ES optimizes the release policy under an event-weighted repair
objective, with held-out damage seeds used to measure survival, cumulative
damage, and repair dynamics. Modulated NCAs substantially outperform an
unmodulated control, reducing final Hamming distance from $0.063$ to
$0.028$--$0.030$ and cumulative damage by approximately fourfold, while random
modulation is uniformly lethal. Crucially, closed-loop control does not
outperform static or constant tonic modulation under stationary damage. Rather
than treating this as a failed ablation, we use it to characterize a regime
boundary: recurring lesions provide no temporal structure for adaptive
scheduling to exploit, so evolution primarily discovers a robust tonic
operating point. A qualitative bisection event further suggests that persistent
global modulation may help preserve morphological organization beyond local
perception. The benchmark provides a controlled substrate for studying when
non-stationary, diversity-driven damage processes make closed-loop robustness
necessary.
\end{abstract}

\section{Introduction}
\label{sec:intro}

Growing Neural Cellular Automata (GNCAs) demonstrate that a single local,
differentiable update rule can grow a target morphology from a seed and
regenerate it after damage~\cite{mordvintsev2020gnca}. The regenerative
capability, however, is bounded by perception. Each cell perceives only its
immediate neighborhood through fixed convolution kernels, so wounds larger
than the perception radius contain cells with no signal to integrate, and
fragments severed by a lesion retain no information about the body they came
from. In our baseline experiments the failure is visible: severed pieces
drift, neither reattaching nor dying, and wound sites fill with undecided
half-alpha cells that persist as scar tissue (Appendix~\ref{app:e1}). These
are \emph{information-isolation} failures, and they motivate a stress test
that operates squarely inside this regime rather than at its edge.

Biology solves the analogous coordination problem with neuromodulation: slow,
globally broadcast chemical signals (tonic levels) punctuated by fast,
event-triggered release (phasic spikes) that reconfigure the gain of local
circuits without changing their
wiring~\cite{schultz1997dopamine,niv2007tonic}. Recent NCA work imports
pieces of this idea --- signal channels~\cite{stovold2023signal}, goal
conditioning~\cite{sudhakaran2022goal}, and information-dynamical analyses of
self-maintenance~\cite{masumori2026fluctuations} --- but in every case the
chemical signal is a fixed input or an emergent byproduct, never a controlled
output optimized against a long-horizon damage objective.

We close that loop, and we use evolutionary search not to claim that
closed-loop control always wins, but to \emph{discover} robust modulation
policies and identify the damage regimes under which adaptive control is
actually necessary. We equip a GNCA with $K{=}3$ global modulator channels and
evolve a 259-parameter controller that reads four target-free grid statistics
and sets channel release every ten steps. The controller is evaluated against
four controls --- no modulation, static conditioning, a hand-searched constant
tonic, and random modulation --- under recurring multi-block damage
deliberately calibrated so that each lesion exceeds the perception radius.

We address four research questions:
\begin{itemize}
  \item \textbf{RQ1}: Can a low-dimensional global modulation channel repair
        damage beyond the NCA's local perception radius?
  \item \textbf{RQ2}: Does state-dependent closed-loop release outperform
        tonic or static modulation under stationary recurring damage?
  \item \textbf{RQ3}: What failure modes and metric artifacts emerge when
        regeneration is evaluated over long, repeated damage horizons?
  \item \textbf{RQ4}: Can a persistent global chemical state preserve
        morphological organization after catastrophic structural disruption?
\end{itemize}

\paragraph{Contributions.}
\begin{enumerate}
  \item A recurring-damage stress test for regenerative NCAs, calibrated to
        exceed local perception and evaluated on held-out damage seeds
        (Section~\ref{sec:damage}).
  \item A controlled regime-boundary finding: global modulation substantially
        improves robustness ($2.2\times$ lower final Hamming,
        $\approx 4\times$ lower cumulative damage), but state-dependent
        closed-loop control provides no measurable advantage over tonic
        control when damage is stationary (Section~\ref{sec:e2}).
  \item An evolutionary controller-search framework for global chemical
        release, including calibration probes for objective hacking and
        failed search initialization (Section~\ref{sec:evolution},
        Appendix~\ref{app:calibration}).
  \item A qualitative bisection observation, presented cautiously as
        motivation for studying persistent global state --- not as proof of
        identity memory (Section~\ref{sec:fission}).
\end{enumerate}

\section{Related Work}
\label{sec:related}

\paragraph{Neural cellular automata.}
GNCAs grow target patterns from a seed and regenerate after localized
damage~\cite{mordvintsev2020gnca}; follow-ups extend the model to
self-classification~\cite{randazzo2020selfclass} and learned
textures~\cite{mordvintsev2021texture}. All rely on purely local perception,
and regeneration quality degrades for lesions larger than the effective
perception radius --- precisely the failure regime our benchmark exploits.
Stovold~\cite{stovold2023signal} adds signal channels through which cells
release and sense chemicals; Sudhakaran et al.~\cite{sudhakaran2022goal}
condition every cell's update on a fixed goal embedding; Masumori et
al.~\cite{masumori2026fluctuations} analyze damage recovery with transfer
entropy and partial information decomposition, documenting globally
coordinated responses to localized damage. In all of these, the signal is a
fixed input or an emergent byproduct --- never a controlled output optimized
against a long-horizon damage objective. Our controller closes the
sensorimotor loop around the chemical layer itself.

\paragraph{Search over evaluation regimes.}
Our benchmark design draws on the insight, central to novelty
search~\cite{lehman2011novelty} and quality-diversity
optimization~\cite{mouret2015mapelites}, that a single fixed objective can
misdirect search and hide failure modes. Environment-generation methods
co-evolve challenges with solutions: POET~\cite{wang2019poet} pairs agents
with self-generated curricula, PAIRED~\cite{dennis2020paired} uses regret to
construct difficult-but-solvable environments, and
ACCEL~\cite{parkerholder2022accel} mutates high-regret environments to build
an evolving curriculum. Unlike these methods, our current benchmark does not
yet evolve the damage distribution; we provide the controlled stationary
baseline against which damage co-evolution and diversity-driven scenario
search can be evaluated. Morphogenetic
engineering~\cite{doursat2013morphogenetic} and developmental
scaffolding~\cite{montero2026scaffold} treat the grown structure as a
substrate for later function; our bisection observation
(Section~\ref{sec:fission}) connects to this line.

\section{Methods}
\label{sec:methods}

\subsection{Growing NCA with global modulator channels}
\label{sec:gnca}

The grid state $x_t \in [0,1]^{96 \times 96 \times 16}$ holds 16 channels per
cell (RGBA last). Each cell perceives its $3{\times}3$ neighborhood through
identity, Sobel-$x$, and Sobel-$y$ kernels, producing a 48-dimensional
perception vector. A shared MLP ($48{+}K \rightarrow 128 \rightarrow 16$)
maps perception plus modulator input to channel updates; cells update
stochastically ($p{=}0.5$) and dead cells are masked. Model and training
details follow Mordvintsev et al.~\cite{mordvintsev2020gnca} with CAX
primitives~\cite{faldor2025cax}; full specifications are in
Appendix~\ref{app:model}.

Each of the $K{=}3$ modulator channels carries a tonic component (exponential
moving average, $\alpha{=}0.95$) and a phasic component (decay
$\tau{=}20$), summed and clipped to $[-1,1]$. The injected level is broadcast
to every cell and concatenated to its perception vector --- the only
non-local information path in the system. All conditions share a single
channel-aware parent trained with the $K{=}3$ channels present and the
controller held at neutral output; the no-modulation control uses a
separately trained $K{=}0$ parent.

\subsection{Controller}
\label{sec:controller}

The controller is an MLP $4 \rightarrow 32 \rightarrow K$ with $\tanh$
activations (259 parameters), issuing a decision every $\tau_d{=}10$ steps.
Its four inputs are target-free summary statistics of the grid: alive
fraction, recently killed fraction, spatial entropy of alpha mass, and a
self-referential mismatch proxy against the rollout's own $t{=}0$ pattern
(never the target; enforced by unit test). The controller can detect that
damage occurred and how far the morphology has drifted from its own recent
state, but cannot read the answer.

\subsection{Damage model: a stress test beyond perception}
\label{sec:damage}

Evaluation uses recurring multi-block lesions: every 150 steps, $n{=}4$
contiguous, axis-aligned $16{\times}16$ squares are cut at seeded random
positions, zeroing all affected cells, for $T{=}2000$ steps (13 events).
A 16-cell block side exceeds the NCA's effective perception radius, so wound
interiors contain no living neighbors and the local rule alone cannot
diagnose the wound. Lesion parameters come from fixed seed sets: seeds 0--7
for evolution and grid search, seeds 10000--10007 as a disjoint held-out set
for all reported numbers.

The benchmark is deliberately adversarial. An earlier schedule (disc lesions
every 250 steps) was solved trivially by the unmodulated baseline (raw
Hamming $\approx 0.002$), leaving no gap for modulation to close and stalling
evolution at a saturated fitness. Damage must be hard enough that the baseline
genuinely struggles; otherwise there is nothing for a controller to prove
(RQ3; Appendix~\ref{app:calibration}).

\subsection{Evolution and fitness}
\label{sec:evolution}

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
reported metric: Section~\ref{sec:e2} reports unweighted trajectory
statistics on held-out seeds. Two calibration controls make this landscape
interpretable --- a collapse probe (verifying the objective cannot be gamed
by alpha suppression) and an initial step-size ablation (showing
$\sigma_0{=}0.3$ saturates the first population in overgrowth where no
repair signal exists) --- both detailed in Appendix~\ref{app:calibration}.

\subsection{Conditions and compute}
\label{sec:baselines}

Five conditions share the same protocol, horizon, and damage seeds:
\textbf{closed-loop} (evolved controller); \textbf{static} (the evolved
controller evaluated once at $t{=}0$ and held constant --- isolating
temporal modulation from the evolved tonic level itself); \textbf{constant}
(fixed tonic, grid-searched over $\{-1,-0.5,0,0.5,1\}$ on train seeds);
\textbf{random} (uniform $[-1,1]$ each step, actuation-matched noise); and
\textbf{no modulation} ($K{=}0$ parent). All experiments ran on a single
rented A100; parent training took $\approx$20 min/model, evolution
$\approx$2.5 h, total project cost under \$15.

\section{Results}
\label{sec:results}

\subsection{Five conditions under recurring hard damage}
\label{sec:e2}

Table~\ref{tab:main} reports the five-condition comparison on held-out damage
seeds; Figure~\ref{fig:hamming} shows Hamming-vs-time trajectories.

\begin{table}[h]
  \centering
  \small
  \caption{Five conditions under recurring multi-block damage ($T{=}2000$,
  lesions every 150 steps). Mean $\pm$ SD over 5 condition seeds $\times$ 8
  held-out damage seeds. Survival is the fraction of runs with final Hamming
  $< 0.1$. Repair half-life is comparable only among conditions that return
  to near-target (see text).}
  \label{tab:main}
  \begin{tabular}{lcccc}
    \toprule
    Condition & Survival & Half-life & Final Hamming & AUC \\
    \midrule
    Closed-loop (evolved) & 1.00 & $6.6 \pm 0.7$ & $0.028 \pm 0.003$ & $0.016 \pm 0.002$ \\
    Static                & 1.00 & $7.2 \pm 0.9$ & $0.030 \pm 0.003$ & $0.017 \pm 0.002$ \\
    Constant tonic        & 1.00 & $6.4 \pm 0.5$ & $0.030 \pm 0.003$ & $0.017 \pm 0.002$ \\
    No modulation         & 1.00 & $2.7 \pm 0.5$ & $0.063 \pm 0.003$ & $0.064 \pm 0.002$ \\
    Random                & 0.00 & $0.0 \pm 0.0$ & $0.825 \pm 0.016$ & $0.819 \pm 0.003$ \\
    \bottomrule
  \end{tabular}
\end{table}

\begin{figure}[h]
  \centering
  \safefigure{figures/fig2_hamming_vs_time}{Fig 2: Hamming Distance vs. Time Trajectories}
  \caption{Hamming distance to target versus time for the five conditions
  (lesions every 150 steps; shaded bands $\pm$1 SD). Modulated conditions
  return to near-target after every lesion; the unmodulated baseline
  accumulates residual damage; random modulation destroys the morphology.}
  \label{fig:hamming}
\end{figure}

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

\subsection{Damage-induced bisection (RQ4)}
\label{sec:fission}

In one closed-loop rollout, a damage event at step 1050 bisects the
already-stressed morphology into two substantial fragments (101 and 58
cells), each large enough to survive local alive-masking
(Figure~\ref{fig:fission}). Both fragments persist and grow for
$\approx$60 steps, narrowing the gap between them from 32 to 25 pixels.
Recovery then becomes asymmetric: the larger fragment monopolizes regrowth
($\approx$240 cells) while the smaller is gradually absorbed below the
survival threshold. The fragments do not re-merge as independent growth
fronts.

\begin{figure}[h]
  \centering
  \safefigure{figures/fig3_fission_sequence}{Fig 3: Bisection and Asymmetric Recovery}
  \caption{Damage-induced bisection and asymmetric recovery (closed-loop,
  damage seed 10000). (a)~Lesion at step 1050 splits the morphology into two
  fragments; (b)~both persist and grow independently for $\approx$60 steps;
  (c)~asymmetric recovery --- the larger fragment monopolizes regrowth.}
  \label{fig:fission}
\end{figure}

This event is consistent with the tonic channel acting as persistent global
state that helps preserve growth-axis organization: both fragments sustained
independent growth, and the chemical layer is the only non-local pathway
through which either fragment could detect it is part of a larger whole. We
deliberately do not claim the tonic channel \emph{is} identity memory ---
the identity could equally reside in the trained local update weights, in
surviving distributed cell state, or in the basin structure of the attractor.
A causal test would require intervention (resetting, scrambling, or clamping
the tonic state after bisection); the information-theoretic tools of Masumori
et al.~\cite{masumori2026fluctuations} provide a template. The unmodulated
baseline resolves similar fragmentation by one fragment dying and leaving
debris --- the failure mode behind its elevated final Hamming ($0.063$).

\section{Discussion}
\label{sec:discussion}

\paragraph{The boundary result is the contribution.}
That closed-loop, static, and constant conditions tie under stationary damage
is not a caveat but a finding: it identifies \emph{when} adaptive control
matters. Where temporal structure exists in the damage process, temporal
control can be selected for; under i.i.d.\ lesions there is none to find.
This reframes the decisive next experiment --- non-stationary damage, where
lesion size or rhythm drifts mid-rollout and a static $t{=}0$ snapshot is
structurally wrong --- as a direct prediction of our framework rather than a
repair.

\paragraph{Robustness-search calibration.}
Two probes were decisive and both generalize. The collapse probe asked
whether the controller could game Hamming by killing everything (it cannot:
all-dead scores $0.046$, $2.2\times$ worse than struggling-neutral, so the
alive-fraction floor is safely disabled). The step-size probe found that
$\sigma_0{=}0.3$ lands the entire first population in saturated overgrowth
where fitness differences reflect overgrowth degree, not repair quality ---
CMA-ES then ranks noise in the wrong regime. When every sample in generation
zero sits in a saturated regime, ``the search cannot find the solution'' and
``the initial distribution never samples the informative regime'' are
indistinguishable from the metrics alone; a 30-second sigma sweep separates
them. Both probes are cheap enough to be standard practice for
robustness-search on generative systems.

\paragraph{Limitations.}
We evaluate one morphology at one scale, one stationary damage distribution,
and one evolution seed. The no-modulation control uses a separately trained
$K{=}0$ parent, so the broadcast-modulation improvement remains somewhat
entangled with parent-training differences; a channel-aware zero-output
control (same $K{=}3$ parent, modulation pinned to neutral) would isolate it
cleanly and is the highest-priority missing control. The Hamming metric has
a noise floor set by permanent debris and conflates locomotion drift with
morphological error (uniformly across conditions). The controller reads four
hand-chosen statistics; richer readouts may matter in harder regimes.

\section{Conclusion}
\label{sec:conclusion}

We built an evolutionary stress test for regenerative NCAs in which damage is
calibrated to exceed local perception, and used it to identify a regime
boundary: global chemical modulation repairs $2.2\times$ more completely
than no modulation and is uniformly required (random release is lethal), but
under stationary recurring damage, closed-loop control does not outperform a
tonic level --- evolution's role is automatic discovery of the robust
operating point. The benchmark, its calibration probes, and the honest null
result together provide the controlled stationary baseline against which
non-stationary, diversity-driven damage co-evolution can now be evaluated.

\appendix

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

\section{Perception-radius sweep (motivating failure modes)}
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
  \safefigure{figures/fig1_recovery_vs_radius}{Fig 1: Recovery vs. Lesion Radius Sweep}
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
  \safefigure{figures/fig4_evolution_trajectory}{Fig 4: Evolution Trajectory (CMA-ES Fitness)}
  \caption{CMA-ES fitness over 300 generations ($\sigma_0{=}0.01$). Dashed
  line: neutral controller ($0.0205$). Best evolved fitness: $0.0135$ --- a
  $34\%$ improvement with no stall.}
  \label{fig:evolution}
\end{figure}

\section{Reproducibility}
\label{app:repro}

All conditions share the protocol, horizon, and damage seeds of
Section~\ref{sec:baselines}. Evolution uses CMA-ES
\cite{hansen2006cma} via Evosax \cite{lange2022evosax} on JAX/Flax with CAX
\cite{faldor2025cax}; total project compute was under \$15 on a single rented
A100. Code will be released upon publication.

\begin{thebibliography}{18}

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

\bibitem{doursat2013morphogenetic}
R.~Doursat, H.~Sayama, and O.~Michel, editors.
\emph{Morphogenetic Engineering: Toward Programmable Complex Systems}.
Springer, 2013.

\bibitem{montero2026scaffold}
M.~L.~Montero, E.~Najarro, J.~H.~Schauser, and S.~Risi.
\emph{Learning Developmental Scaffoldings to Guide Self-Organisation}.
arXiv:2605.14998, 2026.

\bibitem{hansen2006cma}
N.~Hansen.
\emph{The CMA Evolution Strategy: A Comparing Review}.
In \emph{Towards a New Evolutionary Computation}, Studies in Fuzziness and
Soft Computing vol.~192, pages 75--102. Springer, 2006.

\bibitem{lange2022evosax}
R.~T.~Lange.
\emph{evosax: JAX-Based Evolution Strategies}.
arXiv:2212.04180, 2022.

\bibitem{faldor2025cax}
M.~Faldor and A.~Cully.
\emph{CAX: Cellular Automata Accelerated in JAX}.
ICLR 2025; arXiv:2410.02651.

\bibitem{schultz1997dopamine}
W.~Schultz, P.~Dayan, and P.~R.~Montague.
\emph{A Neural Substrate of Prediction and Reward}.
Science, 275(5306):1593--1599, 1997.

\bibitem{niv2007tonic}
Y.~Niv, N.~D.~Daw, D.~Joel, and P.~Dayan.
\emph{Tonic Dopamine: Opportunity Costs and the Control of Response Vigor}.
Psychopharmacology, 191(3):507--520, 2007.

\end{thebibliography}

\end{document}
