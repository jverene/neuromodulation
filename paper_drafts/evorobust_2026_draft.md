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
\vspace*{0.2in}
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

\title{Evolutionary Stress Tests Reveal Parent-Locked Tonic\\Modulation in Neural Cellular Automata}

\begin{document}

\maketitle

\begin{abstract}
We set out to test whether closed-loop control of global chemical signals
makes regenerative Neural Cellular Automata (NCAs) more robust, and built
the evaluation to prove it: recurring multi-block lesions larger than the
perception radius, a small evolved controller for three broadcast
modulator channels, held-out damage seeds. On a single trained model the
experiment looked like a success. Crossing five independently trained
parent seeds --- two independent controller evolutions each, under a
preregistration fixed before data collection --- dissolves that success
into three findings. First, the robustness we had attributed to control is
mostly a property of training: parents trained with channels present beat
unmodulated siblings in five of five seeds (median final-Hamming
reduction $0.008$, up to $\approx 0.14$ on the most fragile parent) with
modulation pinned to neutral. Second, evolved controllers add little on
top: on four parents the effect is absent or noise-level; on the fragile
parent both evolutions reproducibly help ($0.035$--$0.038 \to
0.029$--$0.030$, survival $0.975 \to 1.00$) --- yet its tonic vector is
nearly identical (cosine $0.99$) to a sibling's that does not help, so
the benefit is an interaction with parent dynamics, not a distinctive
operating point. All ten controllers emit flat, nonzero, parent-specific
constants with no lesion-locked response. Third, these constants are
organism-locked: a single-donor probe penalizes five of five siblings
(lethally in two), a full $5{\times}5$ transfer matrix shows most
transfers harmful and none better than the recipient's own controller,
and injecting each donor's tonic constant directly --- no controller at
all --- reproduces its transfer outcomes in 46 of 50 cells, including
every lethal one. Single-parent evaluation would have approved all of it.
The attribution protocol (adversarial damage calibration, hacking and
initialization probes, parent-seed-resolved decomposition) is the
transferable artifact.
\end{abstract}

\section{Introduction}
\label{sec:intro}

Growing Neural Cellular Automata (GNCAs) grow a target morphology from a
single seed and regenerate it after damage using one local, differentiable
update rule~\cite{mordvintsev2020gnca}. The catch is perception: a cell
sees only its immediate neighborhood, so wounds larger than a few cells
contain tissue with no signal to integrate, and a severed fragment has no
way to know what it was part of. The natural remedy is a global signal.
Prior work, our own single-parent study included
(Appendix~\ref{app:singleparent}), adds global modulator channels and
evolves release policies for them --- and reports the outcome on one
trained model.

That single-model habit is the problem this paper is about. When a
modulated NCA outperforms an unmodulated one, three effects are
indistinguishable: the channels being \emph{present during training}, the
controller \emph{acting at run time}, and whatever \emph{transfers} to
another organism. Our single-parent study exhibited
the failure firsthand: it credited closed-loop control for a $2\times$
improvement that a later pilot traced elsewhere --- the evolved
controller transferred lethally to sibling parents while channel parents
with modulation pinned to zero beat the unmodulated baseline in every
seed (Appendix~\ref{app:pilot}). The improvement was real; the
attribution was wrong.

We therefore ran the attribution study that the original design needed:
preregistered, with the primary statistic and decision thresholds fixed
before data collection. Five parent seeds, each trained from scratch as
a $K{=}0$ pair and a channel-aware $K{=}3$ pair; \emph{two} independent
controller evolutions per channel parent; four conditions per seed, all
on held-out damage seeds; then, post-hoc, a full $5{\times}5$ transfer
matrix and a tonic-transplant condition that replaces each controller
with its realized mean output (Section~\ref{sec:design}).
Preregistrations and the mechanical analysis scripts are in the
repository history, timestamped before the runs they govern.

\paragraph{Contributions.}
\begin{enumerate}
  \item A preregistered robustness-attribution protocol for regenerative
        NCAs: adversarial damage calibrated to exceed perception, crossed
        with parent-seed variation, decomposed into
        $E_{\mathrm{train}}$, $E_{\mathrm{ctrl}}$, $E_{\mathrm{transfer}}$,
        with calibration probes that keep the evolution landscape
        interpretable (Appendix~\ref{app:calibration}).
  \item The attribution, settled causally. Training with channels is the
        robust cause (5/5 seeds). The controller effect is parent-
        dependent --- absent or noise-level on four parents, reproducibly
        present on the fragile one --- and the evolved artifact is a
        parent-specific tonic constant, not a policy: it has no
        lesion-locked response, it penalizes every sibling it meets
        (lethally in two of five), and injecting the constant alone
        reproduces the controller's transfer outcomes, every lethal one
        included. The transfer matrix (23/40 off-diagonal cells harmful,
        none better than the recipient's own controller) closes the case.
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
Our design follows a concern shared by novelty
search~\cite{lehman2011novelty} and quality-diversity
optimization~\cite{mouret2015mapelites}: a fixed objective can misdirect
search and hide failure modes. Environment-generation methods co-evolve
challenges with solutions ---
POET~\cite{wang2019poet}, PAIRED~\cite{dennis2020paired},
ACCEL~\cite{parkerholder2022accel}; our benchmark does not yet evolve the
damage distribution and serves as the stationary baseline such
co-evolution can be judged against.

\section{Experimental design}
\label{sec:design}

\paragraph{Damage regime.}
Recurring multi-block lesions --- every 150 steps, $n{=}4$ contiguous
$16{\times}16$ blocks at seeded positions, $T{=}2000$ --- exceed the
perception radius, so wound interiors have no living neighbors
(Appendix~\ref{app:e1}). Damage seeds 0--7 drive evolution; held-out
seeds 10000--10007 drive all reported numbers; the schedule is
deliberately adversarial (Appendix~\ref{app:calibration}).

\paragraph{Parents and controllers.}
Per seed $s$ we train a $K{=}0$ and a channel-aware $K{=}3$ parent from
scratch (Appendix~\ref{app:model}); the $K{=}3$ parent's three global
modulator channels are the only non-local pathway, each carrying a
\emph{tonic} (slow baseline) and \emph{phasic} (event-triggered,
fast-decaying) component --- the distinction from biological
neuromodulation. A 259-parameter controller ($4{\to}32{\to}3$,
$\tanh$) reads four target-free grid statistics and sets release every
10 steps; each parent gets \emph{two} independently evolved controllers
via CMA-ES~\cite{hansen2006cma} in Evosax~\cite{lange2022evosax}
(population 64, $\sigma_0{=}0.01$, 300 generations, different evolution
seeds, event-weighted Hamming objective; Appendix~\ref{app:evolution}).

\paragraph{Preregistered comparison.}
All conditions run on the same held-out damage seeds (5 condition seeds
$\times$ 8 damage seeds) from a shared $t{=}0$ state grown by the $K{=}3$
parent. The primary statistic is
$\Delta_s = H(K{=}3, m{=}0, s) - H(K{=}3, \text{own ctrl}, s)$,
positive when the evolved controller helps \emph{its own} parent. Fixed
before data collection: \emph{controller effect supported} if $\Delta_s >
0$ in $\geq 4/5$ seeds with sign-consistent differences;
\emph{channel-training effect supported} if $H(K0,s) - H(m0,s) > 0$ in
$\geq 4/5$; \emph{transfer failure supported} if the July controller
underperforms the own-controller substantially in $\geq 4/5$. Effects are
reported per seed with median and range; we make no significance claims
at five seeds. The defense study reported here fixed the
two-evolutions-per-parent rule in its preregistration before launch
(\texttt{experiment\_results/20260818\_evoseed\_defense/PREREGISTRATION.md}
in the repository history). Controller-output ($m_t$) series distinguish
tonic calibration from event-locked policy (Appendix~\ref{app:mt}). A
post-hoc follow-up evaluates the full $5{\times}5$ transfer matrix among
the defense parents (both controller replicas, 8 damage $\times$ 3
condition seeds per cell; Table~\ref{tab:matrix}) plus a \emph{tonic
transplant} injecting each donor's realized mean $m_t$ as a constant.

\section{Results}
\label{sec:results}

Table~\ref{tab:effects} reports the preregistered decomposition; the full
per-seed, per-condition numbers behind it are in
Appendix~\ref{app:rawtables}.

\begin{table}[h]
  \centering
  \small
  \caption{Final Hamming (mean over 5 condition seeds $\times$ 8 held-out
  damage seeds) per parent seed: the $K{=}0$ parent; the channel-aware
  $K{=}3$ parent with modulation pinned to zero; and that parent's own
  evolved controller (two independent evolutions; range where the runs
  differ). Survival shown in parentheses where it departs from $1.00$.
  Channel-aware training helps in 5/5 seeds; the controller effect is
  parent-dependent.}
  \label{tab:effects}
  \scriptsize
  \setlength{\tabcolsep}{2.5pt}
  \begin{tabular}{lcccp{1.45in}}
    \toprule
    Parent seed & $K{=}0$ & $K{=}3$, $m{=}0$ & Own ctrl (2 runs) & Interpretation \\
    \midrule
    0 & $0.035$--$0.036$ & $0.029$ & $0.029$--$0.030$ & channel effect; no control effect \\
    1 & $0.163$--$0.175$ (surv.\ $0.05$--$0.08$) & $0.035$--$0.038$ (surv.\ $0.975$) & $0.029$--$0.030$ (surv.\ $1.00$) & channel effect $+$ replicated tonic benefit \\
    2 & $0.030$ & $0.018$--$0.021$ & $0.022$--$0.026$ & channel effect; controller slightly worse \\
    3 & $0.049$--$0.052$ & $0.028$--$0.029$ & $0.032$--$0.033$ & channel effect; controller slightly worse \\
    4 & $0.042$--$0.045$ & $0.033$ & $0.034$--$0.035$ & channel effect; no control effect \\
    \bottomrule
  \end{tabular}
\end{table}

\paragraph{Outcome.}
Applying the preregistered rule: the \emph{channel-training effect is
supported} ($H(K0,s)-H(m0,s) > 0$ in 5/5 seeds; bar was $\geq$4/5); the
\emph{transfer failure is supported} (penalty in 5/5; lethal in 2). The
controller effect is classified as \emph{parent-dependent} ---
preregistered \textbf{Outcome C}: absent or
noise-level on four parents, reproducibly beneficial on the fragile one
(both evolutions: $0.035$--$0.038 \to 0.029$--$0.030$, survival
$0.975 \to 1.00$).

\paragraph{What the controller actually emits.}
All ten evolved controllers emit a \emph{constant at a nonzero level}:
within-rollout per-channel std of $m_t$ is $0.0002$--$0.0046$ --- tonic,
with no lesion-locked response, in 10/10 runs --- while channel means sit
at parent-distinct offsets (Appendix~\ref{app:mt}, Figure~\ref{fig:tonic}). Evolution neither collapses to neutral output nor
discovers a policy: it finds a parent-specific tonic calibration. That
calibration alone does not predict benefit --- seeds 1 and 4 emit nearly
identical vectors (cosine $0.993$), yet only seed 1 benefits --- so the
benefit arises from the interaction between tonic and parent-specific
dynamics, not from a distinctive operating point.

\paragraph{Cross-parent transfer matrix.}
Extending the single-donor probe to the full $5{\times}5$ matrix among
the five defense parents --- both controller replicas per parent, three
condition seeds per cell, 40 off-diagonal cells
(Table~\ref{tab:matrix}, Figure~\ref{fig:matrix}) --- 23/40 transfers
are harmful (survival $<0.9$ or final Hamming worse than the recipient's
zero-output baseline by $>0.005$), 15 are indistinguishable from
zero-output, and 2 beat it. No foreign controller beats the recipient's
own controller beyond noise. Lethal transfer replicates across controller
replicas and condition seeds: donor s3 kills recipient s2 in both
replicas (survival $0.00$); donor s0 on recipient s1 is lethal in e1 and
near-lethal in e2. Classifications are stable across condition seeds
(Figure~\ref{fig:matrix}A).

\paragraph{Tonic alignment structures transfer.}
All lethal instances pair strongly negative tonic-vector cosines
(s3$\to$s2: $-0.61$; s0$\to$s1: $-0.93$), while the only beneficial
transfers --- beating zero-output and matching the own controller ---
occur exclusively within the tonic-aligned pair s4$\to$s1 (cosine
$+0.99$, both replicas; adjacent to the diagonal in
Figure~\ref{fig:matrix}). Across all 40 cells, donor--recipient tonic
cosine correlates with transfer penalty (Pearson $r=-0.30$) and survival
($r=+0.34$); alignment is not sufficient, though --- some benign cells
carry cosines as negative as $-0.95$.

\paragraph{Tonic transplant.}
Injecting each donor's realized mean $m_t$ as a constant in place of its
controller (both replicas, three condition seeds) reproduces the full
controller's outcome in 36/40 off-diagonal cells, including every lethal
transfer (the four mismatches are magnitude differences inside
already-lethal cells), and matches the recipient's own controller in all
10 diagonal cases (46/50 overall; Appendix~\ref{app:rawtables}). The
transferable component is thus the tonic setpoint; the controller's small
dynamic residual did not measurably contribute, on-parent or off.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.84\linewidth]{figures/fig5_transfer_matrix}
  \caption{Transfer redesigned for attribution (replica-averaged; per-cell
  $\Delta$ from the recipient's own controller; survival as the split
  lower triangle; rows/columns cosine-ordered so the aligned pair s4--s1
  borders the diagonal and the anti-aligned pair s3--s2 sits at opposite
  ends). Panel A: controllers. Panel B: the same matrix with each donor's
  tonic constant injected instead --- matching A in 46/50 cells, every
  lethal one included. Values: Table~\ref{tab:matrix}.}
  \label{fig:matrix}
\end{figure}

\section{Discussion}
\label{sec:discussion}

\paragraph{Which component causes robustness.}
Channel-aware training, not run-time control. Parents that grew up with
global channels beat their unmodulated siblings on every seed; the
channels are a developmental scaffold, making the organism more robust
even when they carry no signal at run time. What evolution adds is
narrow: nothing measurable on four parents; on the fragile parent, a
final stabilization that both independent evolutions find --- and since
that parent's tonic vector is nearly identical to a sibling's that gains
nothing from it, the benefit is not a property of the tonic. We
conjecture the fragile parent's dynamics sit near a stability boundary,
so a small tonic push re-stabilizes the attractor (unverified). The broader lesson: held-out \emph{damage}
seeds approved every condition above; crossing \emph{parent} seeds is
what separated training from
control, benefit from harm, and policy from constant.
\paragraph{Why cross-parent transfer fails --- and when it does not.}
The full matrix (Table~\ref{tab:matrix}) sharpens the single-donor
probe: no foreign controller beats the recipient's own controller, most
transfers are harmful (23/40), and the only beneficial ones stay within
the tonic-aligned pair (s4$\to$s1).
The tonic transplant makes it causal: what transfers, for good or ill,
\emph{is} the constant (36/40 off-diagonal, 10/10 diagonal, every lethal
cell included). Each controller's output is a
calibration against its own parent's channel weights, so the same release
level drives different dynamics in a sibling. Evolution found an operating
point for one organism, not a modulation law.

\paragraph{Limitations and future work.}
Five parent seeds, two controller evolutions each; one morphology; one
stationary damage family; the transfer matrix covers only these parents
and replicas. Only one of five parents proved controller-responsive, so
the prevalence of such parents is unknown. Next: multi-parent
(population-based) evolution; non-stationary, diversity-driven damage
co-evolution, for which this benchmark is the controlled baseline.

\paragraph{Conclusion.}
In regenerative NCAs under adversarial recurring damage, robustness comes
from channel-aware training (5/5 seeds, rescuing a nearly lethal
unmodulated parent); evolved modulation is a parent-dependent tonic
calibration --- noise-level on most parents, beneficial on the fragile
one, parent-locked under transfer. Attribute robustness across model
seeds, not only damage seeds.

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
A.~Masumori, H.~Sato, and T.~Ikegami.
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
\emph{Emergent Complexity and Zero-Shot Transfer via Unsupervised Environment Design}.
NeurIPS 2020.

\bibitem{parkerholder2022accel}
J.~Parker-Holder, M.~Jiang, M.~Dennis, M.~Samvelyan, J.~Foerster,
E.~Grefenstette, and T.~Rockt\"aschel.
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

\appendix
\setcounter{table}{0}\setcounter{figure}{0}
\renewcommand{\thetable}{A\arabic{table}}
\renewcommand{\thefigure}{A\arabic{figure}}

\section{Full per-seed, per-condition results}
\label{app:rawtables}

\begin{table}[h]
  \centering
  \small
  \caption{Final Hamming (mean $\pm$ SD, 5 condition seeds $\times$ 8
  held-out damage seeds) and survival for each of the ten runs (5 parent
  seeds $\times$ 2 independent controller evolutions) under hard recurring
  multi-block damage, $T{=}2000$. Conditions: $K{=}3$ parent with
  modulation pinned to zero; the parent's own evolved controller; the
  $K{=}0$ parent.}
  \label{tab:attribution}
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{lcccccc}
    \toprule
    & \multicolumn{2}{c}{$K{=}3$, $m{=}0$} & \multicolumn{2}{c}{$K{=}3$, own ctrl} & \multicolumn{2}{c}{$K{=}0$} \\
    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
    Run & $H$ & surv & $H$ & surv & $H$ & surv \\
    \midrule
    s0\_e1 & $0.0290{\pm}.0135$ & $1.000$ & $0.0303{\pm}.0134$ & $1.000$ & $0.0361{\pm}.0137$ & $1.000$ \\
    s0\_e2 & $0.0294{\pm}.0144$ & $1.000$ & $0.0292{\pm}.0136$ & $1.000$ & $0.0354{\pm}.0109$ & $1.000$ \\
    s1\_e1 & $0.0384{\pm}.0266$ & $0.975$ & $0.0286{\pm}.0177$ & $1.000$ & $0.1628{\pm}.0389$ & $0.075$ \\
    s1\_e2 & $0.0353{\pm}.0256$ & $0.975$ & $0.0301{\pm}.0210$ & $1.000$ & $0.1746{\pm}.0356$ & $0.050$ \\
    s2\_e1 & $0.0183{\pm}.0145$ & $1.000$ & $0.0225{\pm}.0149$ & $1.000$ & $0.0299{\pm}.0178$ & $1.000$ \\
    s2\_e2 & $0.0208{\pm}.0170$ & $1.000$ & $0.0258{\pm}.0161$ & $1.000$ & $0.0296{\pm}.0174$ & $1.000$ \\
    s3\_e1 & $0.0281{\pm}.0177$ & $1.000$ & $0.0332{\pm}.0165$ & $1.000$ & $0.0491{\pm}.0084$ & $1.000$ \\
    s3\_e2 & $0.0289{\pm}.0182$ & $1.000$ & $0.0321{\pm}.0164$ & $1.000$ & $0.0525{\pm}.0136$ & $1.000$ \\
    s4\_e1 & $0.0332{\pm}.0233$ & $1.000$ & $0.0336{\pm}.0223$ & $1.000$ & $0.0420{\pm}.0111$ & $1.000$ \\
    s4\_e2 & $0.0332{\pm}.0225$ & $1.000$ & $0.0345{\pm}.0222$ & $1.000$ & $0.0452{\pm}.0093$ & $1.000$ \\
    \bottomrule
  \end{tabular}
\end{table}

\begin{table}[h]
  \centering
  \small
  \caption{Cross-parent transfer probe (unchanged from the per-parent
  study): the July donor controller evaluated on each recipient parent.
  Final Hamming and survival on held-out damage seeds; transfer is a
  penalty in 5/5 recipients, lethal in 2.}
  \label{tab:transfer}
  \begin{tabular}{lcc}
    \toprule
    Recipient seed & Final Hamming & Survival \\
    \midrule
    0 & $0.036$ & $1.00$ \\
    1 & $0.249$ & $0.00$ (lethal) \\
    2 & $0.434$ & $0.00$ (lethal) \\
    3 & $0.100$ & $0.45$ \\
    4 & $0.041$ & $1.00$ \\
    \bottomrule
  \end{tabular}
\end{table}

\begin{table}[h]
  \centering
  \small
  \caption{Full $5{\times}5$ cross-parent transfer matrix among the five
  defense parents, both controller replicas (top: e1; bottom: e2). Rows:
  donor controller; columns: recipient parent. Cells give mean final
  Hamming / worst-case survival over 3 condition seeds $\times$ 8 held-out
  damage seeds (24 rollouts per cell). Diagonal
  (bold): recipient's own controller (survival $1.00$ throughout).
  $^{\dagger}$lethal (survival $0.00$). Across the 40 off-diagonal cells:
  23 harmful, 15 indistinguishable from zero-output, 2 beneficial ---
  the beneficial cells are exclusively s4$\to$s1, the tonic-aligned pair
  (cosine $+0.99$), in both replicas.}
  \label{tab:matrix}
  \footnotesize
  \setlength{\tabcolsep}{3.5pt}
  \begin{tabular}{lccccc}
    \toprule
    e1 donor & recip.\ s0 & recip.\ s1 & recip.\ s2 & recip.\ s3 & recip.\ s4 \\
    \midrule
    s0 & \textbf{0.0315} & $0.470/0.00^{\dagger}$ & $0.097/0.38$ & $0.028/1.00$ & $0.052/0.88$ \\
    s1 & $0.035/1.00$ & \textbf{0.0306} & $0.023/1.00$ & $0.049/1.00$ & $0.035/1.00$ \\
    s2 & $0.029/1.00$ & $0.043/0.88$ & \textbf{0.0201} & $0.033/1.00$ & $0.092/0.50$ \\
    s3 & $0.035/1.00$ & $0.036/0.88$ & $0.208/0.00^{\dagger}$ & \textbf{0.0295} & $0.044/1.00$ \\
    s4 & $0.034/1.00$ & $0.030/1.00$ & $0.023/1.00$ & $0.044/1.00$ & \textbf{0.0340} \\
    \midrule
    e2 donor & & & & & \\
    \midrule
    s0 & \textbf{0.0291} & $0.108/0.25$ & $0.056/0.88$ & $0.032/1.00$ & $0.039/1.00$ \\
    s1 & $0.032/1.00$ & \textbf{0.0292} & $0.025/1.00$ & $0.049/1.00$ & $0.032/1.00$ \\
    s2 & $0.041/1.00$ & $0.074/0.62$ & \textbf{0.0219} & $0.033/1.00$ & $0.113/0.25$ \\
    s3 & $0.028/1.00$ & $0.058/0.75$ & $0.239/0.00^{\dagger}$ & \textbf{0.0262} & $0.028/1.00$ \\
    s4 & $0.030/1.00$ & $0.030/1.00$ & $0.020/1.00$ & $0.033/1.00$ & \textbf{0.0336} \\
    \bottomrule
  \end{tabular}
\end{table}

\paragraph{Tonic transplant.}
Replacing each donor controller by its realized mean $m_t$ injected as a
constant (same protocol, both replicas, 3 condition seeds) reproduces the
controller's outcome in 36/40 off-diagonal cells and 10/10 diagonal cells
(46/50 overall; Figure~\ref{fig:matrix}B) --- including every lethal
cell, where the transplant kills exactly as the controller does. The
four mismatches are magnitude differences inside already-lethal cells.
Under the tested regime, the controllers' small dynamic residual did not
measurably contribute, on-parent or off (full CSVs in the repository).

AUC mirrors the same ordering in every seed (full CSVs in the repository).

\section{Controller-output ($m_t$) diagnostics}
\label{app:mt}

\begin{table}[h]
  \centering
  \small
  \caption{Tonic output of the evolved controllers: per-channel mean of
  $m_t$, averaged over the two independent evolutions per seed. All ten
  controllers sit at nonzero, parent-distinct offsets; within-rollout
  per-channel std of $m_t$ is $0.0002$--$0.0046$ everywhere --- flat,
  tonic output with no lesion-locked response in 10/10 runs.}
  \label{tab:mt}
  \begin{tabular}{lc}
    \toprule
    Seed & mean$(m)$ per channel (avg.\ of 2 runs) \\
    \midrule
    0 & $-0.021\;\;{+}0.002\;\;{+}0.027$ \\
    1 & $+0.030\;\;{+}0.012\;\;{-}0.026$ \\
    2 & $-0.026\;\;{-}0.032\;\;{-}0.015$ \\
    3 & $+0.001\;\;{+}0.032\;\;{-}0.010$ \\
    4 & $+0.028\;\;{+}0.007\;\;{-}0.025$ \\
    \bottomrule
  \end{tabular}
\end{table}

\begin{figure}[h]
  \centering
  \includegraphics[width=0.98\linewidth]{figures/fig6_tonic_traces}
  \caption{Controller output $m_t$ over a full evaluation rollout, per
  parent seed, both independent evolutions overlaid (solid/dashed), one
  line per modulator channel. After a brief initial transient while the
  perceptual state settles, every controller holds a constant,
  parent-distinct offset through all lesion events --- tonic calibration,
  not an event-locked policy. Note that the two independent evolutions of
  the same parent sometimes settle at slightly different setpoints.}
  \label{fig:tonic}
\end{figure}

The tonic vector alone does not predict benefit: seeds 1 and 4 emit nearly
identical tonic vectors (cosine similarity $0.993$, Euclidean distance
$0.0065$), yet only seed 1 benefits from its controller. The benefit
arises from the interaction between a tonic calibration and
parent-specific local dynamics, not from a distinctive chemical operating
point.

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

Parents train in $\approx$20--30 min each on one A100; the defense study
ran two independent 300-generation controller evolutions per parent (10
total), each $\approx$2.5--3 h on the study A100, for a total
defense-study cost of $\approx$\$25 of rented GPU time. The transfer
matrix is evaluation-only over existing artifacts: 40 off-diagonal
evaluations, $\approx$2 minutes of A100 time per full $5{\times}5$ replica
matrix. Preregistration,
analysis script, per-seed artifacts (controllers,
parents, trajectories, $m_t$ series), and the mechanical outcome output are
in the repository, timestamped before the results. Code will be released
upon publication.


\end{document}
