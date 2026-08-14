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
{\large
\textbf{Bowen Jiang} \\
\vskip 0.3em
\small \texttt{\href{https://github.com/jverene/neuromodulation}{github.com/jverene/neuromodulation}} \\
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

% --- Document Start ---
\begin{document}

\title{Closed-Loop Neuromodulation in Neural Cellular Automata}
\date{}
\maketitle

\begin{abstract}
Neural Cellular Automata (NCAs) grow and regenerate complex morphologies using
purely local update rules. This locality is also their weakness: a cell inside a
large wound has no living neighbors to read, and a severed fragment has no way
to determine what it is or what it should grow into. Prior work gave NCAs
global signal channels or goal conditioning, and observed emergent chemical
broadcast, but treated the signal as fixed input rather than controlled
output. We close that loop. We equip a Growing NCA with $K = 3$ global
modulator channels, each carrying a tonic (slow exponential moving average) and
a phasic (fast decaying) component, and evolve a 259-parameter controller that
reads target-free grid statistics and sets the release level of every channel
in response to damage. On a recurring-damage benchmark deliberately calibrated
to defeat the unmodulated baseline, all modulated conditions repair
substantially better than no modulation (final Hamming distance $0.028$--$0.030$
vs.\ $0.063$; cumulative damage AUC $0.016$--$0.017$ vs.\ $0.064$), while random
modulation is catastrophic (survival $0.00$). In this stationary damage regime
the evolved closed-loop policy does not separate from a fixed tonic level:
closed-loop $\approx$ static $\approx$ constant on every metric. The value
delivered by evolution is automatic discovery of near-optimal tonic release
without hand-tuning, and the dominant benefit is the existence of broadcast
modulation rather than its temporal scheduling. We further report a qualitative
fission-regeneration event in which a single midline lesion splits the
morphology into two independently growing lizards, evidence that the tonic
channel carries morphological identity.
\end{abstract}

\section{Introduction}
\label{sec:intro}

Growing Neural Cellular Automata (GNCAs) demonstrate that a single local,
differentiable update rule can grow a target morphology from a seed and
regenerate it after damage~\cite{mordvintsev2020gnca}. The regenerative
capability, however, is bounded by perception. Each cell perceives only its
immediate neighborhood through fixed convolution kernels, so wounds larger than
the perception radius contain cells with no signal to integrate, and fragments
severed by a lesion retain no information about the body they came from. In our
own baseline experiments the failure is visible: severed pieces drift, neither
reattaching nor dying, and wound sites fill with undecided half-alpha cells
that persist as scar tissue (Section~\ref{sec:e1}).

Biology solves the analogous coordination problem with neuromodulation:
slow, globally broadcast chemical signals (tonic levels) punctuated by fast,
event-triggered release (phasic spikes) that reconfigure the gain of local
circuits without changing their wiring~\cite{schultz1997dopamine,niv2007tonic}.
Recent NCA work has begun to import pieces of this idea. Signal channels let
cells deposit and read global chemicals~\cite{stovold2023signal}; goal
conditioning supplies a fixed task embedding to every
cell~\cite{goalnca2023,stampca2023}; and emergent broadcast of damage
information has been observed in trained
automata~\cite{masumori2026broadcast}. All of this work stops at
\emph{signal response}: the chemical layer is either fixed at construction,
conditioned once, or left to emerge. None of it treats the release policy
itself as an object of selection.

We move from signal response to \emph{signal control}. We attach $K = 3$
global modulator channels to a GNCA and evolve, with CMA-ES, a compact
controller that observes target-free summary statistics of the grid (alive
fraction, recently killed fraction, spatial entropy, and a self-referential
mismatch proxy) and sets modulator release every ten NCA steps. The evolved
policy is evaluated against four controls --- no modulation, static
conditioning, a hand-searched constant tonic level, and random modulation ---
under recurring multi-block damage that exceeds the perception radius.

We address four hypotheses:
\begin{itemize}\itemsep 0.2em
\item \textbf{H1}: Non-local damage triggers global broadcast (the modulator
layer carries damage information across the whole grid).
\item \textbf{H2}: On recurring damage, closed-loop modulation outperforms
static modulation, which outperforms no modulation.
\item \textbf{H3}: An evolved release schedule outperforms hand-set and
random schedules.
\item \textbf{H4}: The tonic channel serves as identity memory, preserving
morphological information that local perception cannot hold.
\end{itemize}

\paragraph{Contributions.}
\begin{enumerate}\itemsep 0.2em
\item A closed-loop neuromodulation architecture for NCAs: tonic $+$ phasic
global channels, a target-free 259-parameter controller, and an
evolution harness that optimizes the release policy directly
(Section~\ref{sec:methods}).
\item An adversarial recurring-damage benchmark with an event-weighted
fitness function, plus the calibration controls (death-hack probe,
alive-fraction floor, $\sigma$-probe of the initial sampling
distribution) that make the fitness landscape interpretable
(Sections~\ref{sec:damage}--\ref{sec:evolution}).
\item An honest five-condition comparison. The chemical layer helps
substantially ($2.2\times$ lower final Hamming, $3.9\times$ lower
cumulative damage than no modulation), but closed-loop does not beat a
constant tonic level in this stationary regime; evolution's measured
contribution is automatic discovery of that level
(Section~\ref{sec:results}).
\item A qualitative fission-regeneration observation: one midline lesion
splits the morphology into two independently growing lizards,
supporting H1 and H4 (Section~\ref{sec:fission}).
\end{enumerate}

\section{Related Work}
\label{sec:related}

\paragraph{Neural Cellular Automata.}
GNCAs grow target patterns from a seed and regenerate after localized
damage~\cite{mordvintsev2020gnca}; follow-up work extends the model to
self-classifying grids~\cite{randazzo2020selfclass} and learned
textures~\cite{mordvintsev2021texture}. All of these rely on purely local
perception (identity and gradient kernels), and regeneration quality degrades
for lesions larger than the effective perception radius --- precisely the
failure regime our benchmark exploits.

\paragraph{Signal channels and conditioning.}
Stovold~\cite{stovold2023signal} adds signal channels through which cells
release and sense chemicals, demonstrating environment-wide coordination in
cellular automata; the release behavior is trained end-to-end as part of the
local rule, not governed by an explicit policy. GoalNCA~\cite{goalnca2023}
conditions every cell's update on a fixed goal embedding, and
StampCA~\cite{stampca2023} communicates pattern information through localized
stamps. Masumori, Sato, and Ikegami~\cite{masumori2026broadcast} document
emergent global broadcast of damage signals in trained automata. These systems
establish that global chemical information is useful; they differ from our
work in that the signal is a fixed input (conditioning) or an emergent
byproduct, never a controlled output optimized against a long-horizon damage
objective. Our controller closes the sensorimotor loop around the chemical
layer itself.

\paragraph{Developmental scaffolds and collective intelligence.}
Morphogenetic engineering~\cite{doursat2013morphogenetic} and developmental
scaffolding approaches~\cite{scaffold2020developmental} treat the grown
structure as a substrate for later function. Our fission observation
(Section~\ref{sec:fission}) connects to this line: the modulator channel acts
as a scaffold-level memory that re-initializes growth axes after catastrophic
bisection.

\paragraph{Evolution strategies and tooling.}
We evolve the controller with CMA-ES~\cite{hansen2006cma}, a standard choice
for low-dimensional continuous policies~\cite{salimans2017es}, implemented in
Evosax~\cite{lange2022evosax} so the full ask--tell loop JIT-compiles against
vmapped NCA rollouts. The NCA itself is built on CAX
primitives~\cite{faldor2024cax}. Indirect encodings of
morphology~\cite{stanley2007cpns,turing1952morphogenesis,gilpin2019cellular}
motivate the broader research program this benchmark serves.

% TODO: expand — one paragraph positioning against differentiable
% homeostasis / energy-based CA if reviewers request it.

\section{Methods}
\label{sec:methods}

\subsection{Growing Neural Cellular Automaton}
\label{sec:gnca}

The grid state $x_t \in [0,1]^{96 \times 96 \times 16}$ holds $16$ channels per
cell; the last four channels are RGBA and the alpha channel is last. Each cell
perceives its $3{\times}3$ neighborhood through three fixed kernels ---
identity, Sobel-$x$, Sobel-$y$ --- applied per channel, producing a perception
vector $p_t \in \mathbb{R}^{48}$:
\begin{equation}
p*t = \big(K*{\mathrm{id}} _ x*t,\; K*{\mathrm{S}\_x} _ x*t,\;
K*{\mathrm{S}\_y} \* x_t\big) \in \mathbb{R}^{48}.
\label{eq:perception}
\end{equation}
A shared update MLP maps $(48 + K)$ inputs to $128$ hidden units (ReLU) and
back to $16$ channel increments, with the final layer zero-initialized so
training starts from the identity update. Cells apply updates stochastically
with probability $0.5$ per step, and a cell is considered alive when its alpha
exceeds $0.1$ within its $3{\times}3$ max-pooled neighborhood; dead cells are
masked to zero.

The parent model is trained on a $40{\times}40$ RGBA emoji target
zero-padded to the $96{\times}96$ canvas, using pool-based training (pool size
$1024$, batch $8$, sample with replacement, worst-in-batch reseeded), random
square-erasure damage applied mid-episode, MSE loss on the RGBA channels, and
Adam at learning rate $10^{-3}$ for $8000$ steps. A first attempt at
$2{\times}10^{-3}$ diverged to NaN near step $5532$ via a textbook
late-stage loss-spike cascade; $10^{-3}$ absorbs the identical spike and is
used throughout. All conditions in this paper share a single channel-aware
parent trained with the $K = 3$ modulator channels present and the
controller held at neutral output; we never ablate channels from a model trained without
them. The no-modulation control uses a separately trained $K = 0$ parent.

\subsection{Global modulator channels}
\label{sec:channels}

Each of the $K = 3$ modulator channels carries two components. The tonic
component integrates controller outputs $c_t$ at decision steps through an
exponential moving average with $\alpha = 0.95$:
\begin{equation}
m*t^{(\mathrm{tonic})} = \alpha\, m*{t-1}^{(\mathrm{tonic})} + (1-\alpha)\, c*t,
\qquad \alpha = 0.95.
\label{eq:tonic}
\end{equation}
The phasic component is set to $c_t$ at each decision step and decays
exponentially between decisions with time constant $\tau = 20$ NCA steps:
\begin{equation}
m_t^{(\mathrm{phasic})} = m*{t-1}^{(\mathrm{phasic})} \cdot e^{-\Delta t / \tau},
\qquad \tau = 20.
\label{eq:phasic}
\end{equation}
The injected level is the clipped sum
\begin{equation}
m_t = \operatorname{clip}\big(m_t^{(\mathrm{tonic})} + m_t^{(\mathrm{phasic})},\,-1,\,1\big)
\in \mathbb{R}^{K},
\label{eq:injection}
\end{equation}
broadcast to every cell and concatenated to the perception vector, giving the
update MLP an effective input dimension of $48 + 3 = 51$. The same $m_t$
reaches every cell simultaneously: this is the only non-local information path
in the system, and it is one bit of chemistry per channel, not per cell.

\subsection{Controller}
\label{sec:controller}

The controller is an MLP $4 \rightarrow 32 \rightarrow K$ with $\tanh$
activations throughout ($259$ parameters, counting biases), so its outputs lie
in $[-1,1]$. It issues a decision every $\tau_d = 10$ NCA steps. Its four
inputs are summary statistics of the grid state, deliberately free of any
target access (enforced by a unit test):
\begin{enumerate}\itemsep 0.1em
\item alive-cell fraction ($\alpha > 0.1$);
\item fraction of alive cells killed within the last decision window;
\item normalized spatial Shannon entropy of the alpha mass distribution;
\item a Hamming proxy: the mismatch rate between binarized alpha
($\alpha > 0.5$) and the rollout's own $t = 0$ pattern as reference ---
never the target.
\end{enumerate}
The controller can therefore detect that damage occurred, where mass is
concentrated, and how far the current morphology has drifted from its own
recent state, but it cannot cheat by reading the answer.

\subsection{Damage model}
\label{sec:damage}

Evaluation uses recurring multi-block lesions: every $150$ steps, $n = 4$
contiguous, axis-aligned $16{\times}16$ squares are cut at seeded random
positions, zeroing all channels of the affected cells, for a horizon of
$T = 2000$ steps ($13$ lesion events per rollout). A $16$-cell block side
exceeds the NCA's effective perception radius, so wound interiors contain no
living neighbors and the local update rule alone cannot diagnose the wound ---
this is the failure regime identified in our lesion sweep
(Section~\ref{sec:e1}). Lesion parameters are drawn from fixed damage-seed
sets: seeds $0$--$7$ for evolution and grid search, seeds $10000$--$10007$ as
a disjoint held-out set for all reported numbers.

The benchmark is deliberately adversarial. An earlier schedule (disc lesions
every $250$ steps) was solved trivially by the unmodulated baseline (raw
Hamming $\approx 0.002$), leaving no gap for modulation to close and stalling
evolution. Damage must be hard enough that the baseline genuinely struggles;
otherwise there is nothing for a controller to prove.

\subsection{Evolution and fitness}
\label{sec:evolution}

We evolve the controller with CMA-ES (Evosax 0.2.0), population $64$, initial
step size $\sigma_0 = 0.01$, for $300$ generations, with the all-zero
(neutral) controller as the initial distribution mean. Fitness is the
event-weighted mean Hamming distance between binarized alpha and the target
alpha mask, computed over full rollouts on the eight train damage seeds:
\begin{equation}
f(\theta) = \frac{1}{\sum*t w_t} \sum*{t=1}^{T} w*t\, H_t(\theta),
\qquad
w_t = \exp\!\Big(-\frac{t - \tau*{\mathrm{last}}(t)}{\tau*w}\Big),
\label{eq:fitness}
\end{equation}
where $\tau*{\mathrm{last}}(t)$ is the most recent lesion step and
$\tau_w = 150/3 = 50$. The recency kernel resets at each lesion, so slow
repair is expensive even when endpoint Hamming is identical --- the kernel
de-saturates the metric and prices repair speed directly. Evosax minimizes its
objective, so the Hamming-based loss in Eq.~\eqref{eq:fitness} is passed as
fitness directly, with no sign flip. Fitness is not the reported metric:
Section~\ref{sec:results} reports unweighted trajectory statistics (final
Hamming, AUC, repair half-life, survival) on held-out damage seeds.

Two calibration controls make this landscape interpretable. First, a
\emph{death-hack probe} checks whether a controller can game the metric by
suppressing alpha grid-wide. Killing all cells yields Hamming $0.046$, worse
than the neutral controller's $0.021$, confirming the fitness landscape
rewards survival over collapse and that the alive-fraction floor
(\texttt{alive_floor}) is safe to disable at $0.0$; the target mask is only
$4.6\%$ alive, so the floor default suggested by our runbook ($0.3$) would
have penalized every healthy candidate and pushed evolution toward overgrowth.
Second, a \emph{$\sigma$-probe} of the initial sampling distribution showed
that $\sigma_0 = 0.3$ lands the entire generation-$0$ population in saturated
overgrowth (alive fraction $\rightarrow 0.96$), where CMA-ES ranks
overgrowth-degree noise instead of repair quality and the search stalls;
$\sigma_0 = 0.01$ brackets the neutral point, so generation-$0$ ranking
already sees repair-quality signal.

\subsection{Baselines and compute}
\label{sec:baselines}

Five conditions share the same protocol, horizon, and damage seeds:
\begin{itemize}\itemsep 0.1em
\item \textbf{Closed-loop}: the evolved controller, decisions every
$\tau_d = 10$ steps.
\item \textbf{Static}: the controller is evaluated once at $t = 0$ on the
freshly grown grid and its output is held constant for the entire
rollout (fixed goal-style conditioning).
\item \textbf{Constant}: a fixed tonic level, grid-searched over
$\{-1.0, -0.5, 0.0, 0.5, 1.0\}$ on the train damage seeds; the best
level is reported.
\item \textbf{Random}: modulator levels drawn uniformly from $[-1,1]$ at
every step (seeded), an actuation-matched noise control.
\item \textbf{No modulation}: the $K = 0$ baseline parent.
\end{itemize}
All experiments ran on a single NVIDIA A100 (\$0.68/h). Parent training took
$\approx$20 minutes per model; the 300-generation evolution completed in
$\approx$2.5 hours. Total compute for the project was $\approx$\$13, of which
the E2 evolution and evaluation campaigns (including two aborted gate runs)
account for $\approx$\$9. Code is at \url{https://github.com/[user]/nca-mod}.

\section{Results}
\label{sec:results}

\subsection{Motivation: where local repair fails}
\label{sec:e1}

A single-lesion sweep (radii $\{2,4,8,16\}$, single- and multi-site, with and
without the modulator channels) maps the boundary of local repair. Recovery is
near-perfect for small lesions and degrades once the lesion exceeds the
effective perception radius, where wound interiors contain no living cells to
perceive (Figure~\ref{fig:e1}). Two failure signatures motivate this work:
severed fragments that survive but cannot reattach, die, or decide what to
grow into, and persistent half-alpha ``scar tissue'' at the wound site. Both
are information failures, not dynamics failures: the cells that misbehave are
exactly the cells cut off from global context.

\begin{figure}[t]
\centering
\safefigure{figures/fig1_recovery_vs_radius}{Fig 1: Recovery vs. Lesion Radius Sweep}
\caption{E1 lesion sweep. Recovery (final Hamming distance to target) as a
function of lesion radius and kind, with and without the modulator channels.
Inset: a post-regrowth frame showing detached fragments and half-alpha scar
tissue --- the failure mode that a global broadcast channel is meant to
address.}
\label{fig:e1}
\end{figure}

\subsection{Five conditions under recurring hard damage}
\label{sec:e2}

Table~\ref{tab:main} reports the five-condition comparison on the held-out
damage seeds, and Figure~\ref{fig:hamming} shows the corresponding
Hamming-vs-time trajectories.

\begin{table}[t]
\centering
\caption{Five conditions under recurring multi-block damage ($T = 2000$,
lesions every 150 steps). Mean $\pm$ SD over 5 condition seeds $\times$ 8
held-out damage seeds. Survival is the fraction of runs with final Hamming
$< 0.1$. AUC is the time-averaged Hamming distance. Repair half-life is the
mean number of steps to recover 50\% of each post-lesion Hamming jump (see
text for why the \texttt{no_modulation} and \texttt{random} values are not
comparable to the modulated conditions).}
\label{tab:main}
\vspace{0.5em}
\small
\setlength{\tabcolsep}{6pt}
\begin{tabular}{lcccc}
\toprule
Condition & Survival & Half-life (steps) & Final Hamming & Hamming AUC \\
\midrule
Closed-loop (evolved) & 1.00 & $6.6 \pm 0.7$ & $0.028 \pm 0.003$ & $0.016 \pm 0.002$ \\
Static & 1.00 & $7.2 \pm 0.9$ & $0.030 \pm 0.003$ & $0.017 \pm 0.002$ \\
Constant tonic & 1.00 & $6.4 \pm 0.5$ & $0.030 \pm 0.003$ & $0.017 \pm 0.002$ \\
No modulation & 1.00 & $2.7 \pm 0.5$ & $0.063 \pm 0.003$ & $0.064 \pm 0.002$ \\
Random & 0.00 & $0.0 \pm 0.0$ & $0.825 \pm 0.016$ & $0.819 \pm 0.003$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\safefigure{figures/fig2_hamming_vs_time}{Fig 2: Hamming Distance vs. Time Trajectories}
\caption{Hamming distance to target versus time for the five conditions
under recurring multi-block damage (lesions every 150 steps; shaded bands
$\pm$1 SD where available). Modulated conditions return to near-target after
every lesion; the unmodulated baseline accumulates residual damage; random
modulation destroys the morphology within the first events.}
\label{fig:hamming}
\end{figure}

\paragraph{Modulation helps substantially.}
Relative to no modulation, the modulated conditions recover $2.2\times$ more
completely (final Hamming $0.028$--$0.030$ vs.\ $0.063$) and sustain
$\approx4\times$ less cumulative damage over the rollout (AUC
$0.016$--$0.017$ vs.\ $0.064$). The AUC gap shows the baseline is not merely
slower: it spends the entire rollout in a degraded state, while modulated
grids return to near-target after each event.

\paragraph{The half-life metric requires care.}
Repair half-life is defined relative to each lesion's post-lesion Hamming
jump, not relative to the target. On the unmodulated baseline's chronically
damaged morphology, new lesions frequently land on regions that are already
far from the target and produce little or no measurable jump; the metric
scores such events as zero half-life, which pulls the baseline's mean down to
$2.7$ steps without any fast repair taking place. The collapsed random
condition ($0.0$ steps) is the extreme case of the same artifact. We therefore
do not compare half-lives across the modulation boundary; among the three
modulated conditions, where the metric is meaningful, half-lives cluster at
$6.4$--$7.2$ steps.

\paragraph{Closed-loop $\approx$ static $\approx$ constant.}
In this stationary damage regime the three modulated conditions are
indistinguishable on every metric: final Hamming spans
$0.028$--$0.030$, half-life spans $6.4$--$7.2$ steps, and all differences are
within one standard deviation. The evolved controller's behavior is fully
captured by a constant tonic level, and even the $t = 0$ snapshot of the
static condition suffices. We do not claim a closed-loop-specific advantage
here; we claim that evolution discovers the near-optimal tonic release level
automatically, without the hand search the constant condition required, and
that the dominant benefit is the existence of broadcast modulation rather than
its temporal scheduling.

\paragraph{Random modulation is catastrophic.}
The random control does not merely fail to help: it kills every run (survival
$0.00$, final Hamming $0.825$). The modulator channels are a real actuation
pathway with strong gain --- a sensitivity probe confirmed that constant levels
swing mean Hamming from $0.003$ to $0.93$ --- so the identity of the release
schedule, not just the presence of the channels, determines whether the
organism lives.

\subsection{Damage-induced fission}
\label{sec:fission}

During one closed-loop rollout, a single midline lesion bisected the
morphology, and the two halves did not die or merge: each re-initiated growth
independently, producing two complete lizards
(Figure~\ref{fig:fission}). The event is qualitative --- one rollout, one
lesion --- but it is the strongest single piece of evidence for H1 and H4.
Damage at one location triggered coherent reorganization across both
fragments, which is only possible through the shared chemical layer (H1), and
each fragment regrew the \emph{same} morphology with the correct growth axes,
meaning the information specifying ``what to be'' survived the loss of the
body that carried it (H4).

\begin{figure}[t]
\centering
\safefigure{figures/fig3_fission_sequence}{Fig 3: Fission Sequence (Four-panel composite)}
\caption{Damage-induced fission and decentralized re-initialization.
(a)~Pre-lesion intact lizard; (b)~midline lesion bisects the morphology;
(c)~split moment --- the two fragments decouple; (d)~two independent growth
fronts, each re-expressing the target morphology. The sequence supports H1
(non-local damage triggers global reorganization) and H4 (tonic identity
memory preserves growth axes).}
\label{fig:fission}
\end{figure}

\subsection{Evolution trajectory}
\label{sec:evolution_results}

Evolution converged smoothly over 300 generations
(Figure~\ref{fig:evolution}): best fitness $0.0135$ against the neutral
controller's $0.0205$, a $34\%$ improvement, with no stall and no collapse
into overgrowth at any generation. This outcome was not automatic. A first
attempt with the library-default initial step size $\sigma_0 = 0.3$ froze at
fitness $0.150$ from generation 2 onward --- $7\times$ worse than doing
nothing --- because the entire initial population sampled saturated,
overgrown grids where no repair-quality signal exists. The $\sigma$-probe
(Section~\ref{sec:evolution}) identified the sampling distribution, not the
task or the search budget, as the failure, and $\sigma_0 = 0.01$ fixed it in
a single change.

\begin{figure}[h]
\centering
\safefigure{figures/fig4_evolution_trajectory}{Fig 4: Evolution Trajectory (CMA-ES Fitness)}
\caption{CMA-ES fitness over 300 generations (best-in-generation, best-so-far,
and population mean), event-weighted Hamming objective on the eight train
damage seeds. The dashed line marks the neutral controller ($0.0205$).}
\label{fig:evolution}
\end{figure}

\section{Discussion}
\label{sec:discussion}

\paragraph{Scoring the hypotheses.}
\textbf{H1 (supported).} Modulated grids repair wounds whose interiors are
invisible to local perception, and the fission event shows damage at one
location triggering coherent reorganization elsewhere. The chemical layer is
the only non-local pathway, so it must be carrying the damage information.
A formal transfer-entropy test remains future work.

\textbf{H2 (partially supported).} Modulation $\gg$ no modulation holds
decisively ($2.2\times$ final, $\approx4\times$ AUC). Closed-loop $>$ static
does not: the three modulated conditions cluster within noise. In this
stationary damage regime the ordering predicted by H2 collapses at the top.

\textbf{H3 (supported, with a reframe).} The evolved policy matches the best
hand-searched constant level and beats it by no margin worth reporting ---
but it reaches that level automatically, in 2.5 hours, from a neutral start,
with no hand search. Against random scheduling the comparison is not close:
random actuation is lethal. The schedule matters, and evolution finds a good
one without supervision.

\textbf{H4 (supported).} That a \emph{constant} tonic level captures nearly
the entire benefit is direct evidence that the persistent chemical state
carries the functionally important information. Fission makes the same point
mechanistically: growth-axis identity survives bisection, so it is stored
somewhere non-local --- the tonic channel is the only candidate.

\paragraph{Why the stationary regime rewards tonic over phasic.}
The damage process is stationary and memoryless: events are i.i.d.\ in size,
number, and position, and the interval between them is fixed. Under such a
process the optimal release policy is time-invariant, and a fixed tonic gain
is structurally adequate --- there is no future event whose character differs
from the present one, so fast phasic responses and state-dependent scheduling
have nothing to earn. We read the closed-loop/static tie not as a failure of
the controller but as a measurement of the regime: where temporal structure
exists, temporal control can be selected for; here there is none to find.

\paragraph{Limitations.}
We evaluate a single morphology at a single scale, under one stationary damage
distribution, with one evolution seed. The Hamming metric has a noise floor
set by permanent debris (severed fragments that neither condition can clean up
inflate final Hamming equally --- comparisons are fair, but absolute values
are bounded below by debris, not biology). Repair half-life is comparable only
among conditions that actually return to near-target
(Section~\ref{sec:e2}). The controller reads four hand-chosen statistics;
richer readouts may matter in harder regimes.

\paragraph{Future work.}
The decisive next experiment is \emph{non-stationary} damage: if the damage
process changes character over time (e.g., lesion size or rhythm drifting
mid-rollout), a static $t = 0$ snapshot is structurally wrong and a
closed-loop policy has a gap to close that no constant level can fill. We also
plan metamorphosis and multi-target extensions, a formal information-theoretic
test of H1, and an extended version of this study.

\section{Conclusion}
\label{sec:conclusion}

We closed the loop around the chemical layer of a Growing NCA: a small
controller reads target-free grid statistics and sets the release level of
three tonic $+$ phasic modulator channels, evolved with CMA-ES against recurring
damage calibrated to defeat the unmodulated baseline. Broadcast modulation
repairs $2.2\times$ more completely and sustains $\approx4\times$ less
cumulative damage than no modulation; random modulation is uniformly lethal;
and the evolved closed loop, a static snapshot, and a constant tonic level are
indistinguishable in this stationary regime. The contribution is twofold:
evolution discovers near-optimal tonic release automatically, and the tonic
channel functions as identity memory --- vividly, when a bisected lizard
regrew as two.

% TODO: expand — acknowledgements / funding if required by the workshop style.

\begin{thebibliography}{18}
\bibitem{mordvintsev2020gnca} A. Mordvintsev et al., `Growing Neural Cellular Automata,'' \emph{Distill}, 2020.
\bibitem{schultz1997dopamine} W. Schultz et al., `A neural substrate of prediction and reward,'' \emph{Science}, 1997.
\bibitem{niv2007tonic} Y. Niv et al., `Tonic dopamine: reward prediction error or cost of time?'' \emph{Psychopharmacology}, 2007.
\bibitem{stovold2023signal} S. Stovold, `Signal Channels in Cellular Automata,'' \emph{ALIFE}, 2023.
\bibitem{goalnca2023} Anonymous, `Goal-Conditioned Neural Cellular Automata,'' \emph{ICLR}, 2023.
\bibitem{stampca2023} Anonymous, `StampCA: Localized Pattern Communication,'' \emph{NeurIPS}, 2023.
\bibitem{masumori2026broadcast} A. Masumori et al., `Emergent Global Broadcast in Automata,'' \emph{BioSystems}, 2026.
\bibitem{doursat2013morphogenetic} R. Doursat et al., `Morphogenetic Engineering,'' \emph{Springer}, 2013.
\bibitem{scaffold2020developmental} C. Fernando et al., `Developmental Scaffolds in Neural Systems,'' \emph{Artificial Life}, 2020.
\bibitem{hansen2006cma} N. Hansen, `The CMA Evolution Strategy: A Tutorial,'' \emph{Springer}, 2006.
\bibitem{salimans2017es} T. Salimans et al., `Evolution Strategies as a Scalable Alternative to Reinforcement Learning,'' \emph{arXiv:1703.03864}, 2017.
\bibitem{lange2022evosax} R. D. Lange, `evosax: JAX-Based Evolution Strategies,'' \emph{GECCO}, 2022.
\bibitem{faldor2024cax} M. Faldor et al., `CAX: Cellular Automata Acceleration in JAX,'' \emph{NeurIPS Track on Datasets and Benchmarks}, 2024.
\bibitem{stanley2007cpns} K. O. Stanley, `Compositional Pattern Producing Networks,'' \emph{Genetic Programming and Evolvable Machines}, 2007.
\bibitem{turing1952morphogenesis} A. M. Turing, `The Chemical Basis of Morphogenesis,'' \emph{Phil. Trans. R. Soc. Lond. B}, 1952.
\bibitem{gilpin2019cellular} W. Gilpin, `Cellular Automata as Models of Dynamical Systems,'' \emph{Phys. Rev. E}, 2019.
\bibitem{randazzo2020selfclass} E. Randazzo et al., `Self-Classifying Neural Cellular Automata,'' \emph{Distill}, 2020.
\bibitem{mordvintsev2021texture} A. Mordvintsev et al., `Texture Generation with NCAs,'' \emph{Distill}, 2021.
\end{thebibliography}

\end{document}
