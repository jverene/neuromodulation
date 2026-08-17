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
% TODO(numbers): every [X] filled from analyze_per_parent.py once data lands.
Regenerative Neural Cellular Automata (NCAs) are usually evaluated on a
single trained model, so robustness claims cannot separate what
\emph{training} provides from what run-time \emph{control} provides. We
cross five independently trained parent seeds with four conditions under
recurring multi-block damage that exceeds the perception radius: a $K{=}0$
parent; the channel-aware $K{=}3$ parent with modulation pinned to neutral;
that parent with a controller evolved \emph{for it}; and a cross-parent
transfer probe. The comparison is preregistered: the primary statistic
$\Delta_s = H(\text{zero-output}, s) - H(\text{own controller}, s)$ is
evaluated per seed under fixed criteria. Across five parent seeds,
channel-aware training [X: E\_train summary], evolved controllers [X:
E\_ctrl summary], and the transfer probe [X: E\_transfer summary].
Controller-output diagnostics show [X: tonic calibration / event-locked
policy]. Single-parent evaluation can therefore overattribute robustness
to closed-loop control that is in fact [X: parent-locked / unnecessary /
parent-dependent]. The attribution protocol --- adversarial damage
calibration, objective-hacking and search-initialization probes, and
parent-seed-resolved decomposition --- transfers to other self-organizing
systems.
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
  \item The attribution result: [X: outcome A/B/C one-liner], with
        controller-output diagnostics showing [X: tonic calibration /
        event-locked policy] and a transfer probe showing evolved
        controllers are [X: parent-locked / ---].
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

\paragraph{Damage regime.}
Recurring multi-block lesions: every 150 steps, $n{=}4$ contiguous
$16{\times}16$ blocks are cut at seeded positions, for $T{=}2000$ steps.
A 16-cell side exceeds the perception radius, so wound interiors contain
no living neighbors --- the information-isolation regime mapped by our
lesion sweep (Appendix~\ref{app:e1}). Damage seeds 0--7 drive evolution;
held-out seeds 10000--10007 drive every reported number. The schedule is
deliberately adversarial: a milder disc schedule was solved trivially by
the unmodulated baseline (Appendix~\ref{app:calibration}).

\paragraph{Parents and controllers.}
Per seed $s$: a $K{=}0$ and a channel-aware $K{=}3$ parent, trained from
scratch (8000 steps, LR $10^{-3}$; Appendix~\ref{app:model}). The $K{=}3$
parent carries three global modulator channels --- the only non-local
pathway. A 259-parameter controller ($4{\to}32{\to}3$, $\tanh$) reads four
target-free grid statistics and sets release every 10 steps. For each
parent we evolve its own controller with CMA-ES~\cite{hansen2006cma} via
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
calibration from event-locked policy (Section~\ref{sec:mt}).

\section{Results}
\label{sec:results}

\subsection{Four conditions across five parent seeds}
\label{sec:main}

\begin{table}[h]
  \centering
  \small
  \caption{Final Hamming (mean $\pm$ SD, 5 condition seeds $\times$ 8
  held-out damage seeds) across five parent seeds.}
  \label{tab:attribution}
  \begin{tabular}{lcccc}
    \toprule
    Parent seed & $K{=}0$ & $K{=}3$, $m{=}0$ & $K{=}3$, own ctrl & $K{=}3$, July ctrl \\
    \midrule
    0 & [X] & [X] & [X] & [X] \\
    1 & [X] & [X] & [X] & [X] \\
    2 & [X] & [X] & [X] & [X] \\
    3 & [X] & [X] & [X] & [X] \\
    4 & [X] & [X] & [X] & [X] \\
    \midrule
    median & [X] & [X] & [X] & [X] \\
    \bottomrule
  \end{tabular}
\end{table}

\subsection{Effect decomposition}
\label{sec:effects}

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
    0 & [X] & [X] & [X] \\
    1 & [X] & [X] & [X] \\
    2 & [X] & [X] & [X] \\
    3 & [X] & [X] & [X] \\
    4 & [X] & [X] & [X] \\
    \midrule
    median [range] & [X] & [X] & [X] \\
    \bottomrule
  \end{tabular}
\end{table}

\paragraph{Outcome.}
[X: apply the preregistered rule mechanically; state outcome A/B/C using
the pre-committed wording verbatim.]

\subsection{What the controller actually emits}
\label{sec:mt}

[X: $m_t$ diagnostics per controller --- within-rollout std, correlation
with a post-lesion indicator, $|\Delta m|$ at lesion steps vs baseline
drift --- with the pre-committed wording: \emph{parent-specific tonic
calibration} / \emph{event-locked release policy} / \emph{state-dependent
policy, scheduling unnecessary in this regime}.]

\section{Discussion}
\label{sec:discussion}

\paragraph{Which component causes robustness.}
[X: outcome-dependent, per preregistration. A: channel-aware training
improves baseline robustness and evolution further discovers useful
parent-specific modulation; the policy is parent-locked and can fail
catastrophically under transfer. B: the robustness gain comes primarily
from channel-aware parent training; single-parent controller evaluation
can overattribute robustness to closed-loop control. C: controller
efficacy is parent-dependent; evolution can exploit idiosyncratic parent
dynamics, but neither the learned policy nor its benefit reliably
transfers.]

\paragraph{Why cross-parent transfer fails.}
The transfer probe uses one donor controller, five recipients --- a probe,
not a full transfer study (which would test every donor--recipient pair).
Even so, the observed [X: lethal transfer] is structural: each
controller's output is calibrated against its own parent's channel
weights, so the same release level drives different dynamics in a sibling.
Evolution found [X: an operating point / a policy] for one organism, not a
modulation law.

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

[X: two to three sentences matching the outcome.] The broader contribution
is methodological: robustness claims about self-organizing systems should
be attributed across model seeds, not only damage seeds.

\appendix

\section{Pilot: single-parent evaluation hides parent-locking}
\label{app:pilot}

A three-parent pilot (2026-08-16) first exposed the confound. The
single-parent controller from our original five-condition study
transferred lethally to two of three sibling parents (survival $0.00$,
final Hamming $0.249$ and $0.434$), while zero-output channel parents beat
the $K{=}0$ baseline in $3/3$ seeds (final Hamming $0.029/0.034/0.020$ vs
$0.034/0.177/0.028$). The original table's headline gap was therefore part
parent-training effect and part parent-seed luck --- motivating the full
attribution study.

\section{Model and training details}
\label{app:model}
[X: carry over unchanged from commit 7da8810 --- perception equation,
update MLP, tonic/phasic equations, parent training protocol, LR
divergence note.]

\section{Perception-radius sweep}
\label{app:e1}
[X: carry over --- fig1 caption and information-isolation failure modes.]

\section{Calibration probes}
\label{app:calibration}
[X: carry over --- collapse probe, initial step-size probe, adversarial
schedule motivation, fig4.]

\section{Evolution objective}
\label{app:evolution}
[X: carry over --- event-weighted fitness equation and de-saturation
rationale.]

\section{Single-parent five-condition study}
\label{app:singleparent}
[X: carry over --- original Table 1 and fig2, reframed as the single-parent
result whose attribution the main text resolves; the five conditions and
their indistinguishability.]

\section{Reproducibility}
\label{app:repro}
[X: seeds and protocol, preregistration + analysis scripts in repository
history timestamped before results, compute cost, code release note.]

\begin{thebibliography}{18}
[X: carry over the 18 verified entries unchanged from commit 7da8810]
\end{thebibliography}

\end{document}
