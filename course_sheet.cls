% Course exercise sheet class
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{course_sheet}[2025/10/07 Course exercise sheet class]

\LoadClass[11pt]{article}

% Packages and layout
% Top margin: larger overall; title page first line kept at same position via \vspace* in \@maketitle
\RequirePackage[margin=1in,top=1.6cm]{geometry}
\RequirePackage{enumitem}
\RequirePackage{hyperref}
\RequirePackage{xcolor}
\hypersetup{
    colorlinks=true,
    urlcolor=[HTML]{17365C}
}
\RequirePackage{titling}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{environ}
\RequirePackage{graphicx}
\RequirePackage{array}
\RequirePackage{multirow}
\RequirePackage{booktabs}
\RequirePackage{mdframed}
\RequirePackage{longtable}

% Solutions toggle: enabled if \def\solutions{1} is provided
\newif\ifsolutions
\ifdefined\solutions\solutionstrue\fi

% Sheet identifier provided by build; provide a readable default
\providecommand{\sheetname}{sheet_XX}

% Due date storage
\makeatletter
\def\@duedate{TBD}
\def\@published{TBD}
\newcommand{\duedate}[1]{\gdef\@duedate{#1}}
\newcommand{\published}[1]{\gdef\@published{#1}}
\makeatother

% Headers helpers
\newcommand{\classwork}{\section*{Classwork}}
\newcommand{\homework}[1][]{\section*{Homework\hfill#1}}
\newcommand{\miniexam}{%
    \section*{Mini-exam}%
    \medskip\noindent\small\emph{In-class written test; duration and permitted aids follow the announcement in the lecture.}%
    \par\medskip
}
\newcommand{\mockexam}{%
    \section*{Mock Exam}%
    \medskip\noindent\small\emph{Practice questions covering Chapters~1--7 of Cahn and Goldhaber's open-access second edition. Each chapter contains five conceptual-understanding questions, three experimental-analysis/design questions, and two short calculations.}%
    \par\medskip
}

% Problems directory (relative to sandbox working dir)
\providecommand{\problemsdir}{../problems}

% Settings command to load course information
\newcommand{\settings}[1]{\input{../tex/#1.tex}}

% Continuous problem numbering across the sheet
\newcounter{problem}
\renewcommand{\theproblem}{\arabic{problem}}

% Helper to typeset points only when provided
\newcommand{\pointsparen}[1]{\ifx\\#1\\\else\,\textbf{#1\,points}\fi}

% Command to typeset points in brackets with correct abbreviation (dark green, bold)
\newcommand{\pts}[1]{%
    \textcolor[rgb]{0,0.4,0}{\textbf{%
        (#1\,pts)%
    }}%
}

% Warning command for adding disclaimer statements
\newcommand{\warning}{%
    \begin{mdframed}[backgroundcolor=gray!10,linewidth=0pt]
        \scriptsize
        \setlength{\parskip}{0.1em}
        \linespread{0.8}\selectfont
        \textbf{WARNING:} The homework problems are open-ended research tasks. Students are encouraged to use any tools or methods they find helpful to reach a solution. The organizers provide no warranty for the absence of confusion and reserve the right to be amazed by your creativity (or lack thereof).
    \end{mdframed}
}

% \problem[points]{name}{title}: inputs Problems/<name>.tex and prints a header
\newcommand{\problem}[3][]{%
    \refstepcounter{problem}%
    \par\bigskip\noindent\textbf{Problem~\theproblem}:~#3\hfill\pointsparen{#1}\par\medskip
    \begingroup\input{\problemsdir/#2.tex}\endgroup
}

% \problemwithoutpoints{name}{title}: inputs Problems/<name>.tex and prints a header
\newcommand{\problemwithoutpoints}[2]{%
    \refstepcounter{problem}%
    \par\bigskip\noindent\textbf{Problem~\theproblem}:~#2\par\medskip
    \begingroup\input{\problemsdir/#1.tex}\endgroup
}

% Alias to accept \probem as a synonym (typo-tolerant)
\let\probem\problem

% Solution environment: only typeset in solutions builds
\ifsolutions
    \newenvironment{solution}{%
        \par\vspace{0.5em}%
        \noindent\rule{\textwidth}{1pt}\\[0.3em]%
        \noindent\textbf{Solution.}\par\medskip%
        \begingroup%
    }{%
        \endgroup%
        \par\vspace{0.3em}%
        \noindent\rule{\textwidth}{1pt}%
        \par\vspace{0.5em}%
    }
\else
    \NewEnviron{solution}{}
\fi

% Load course information (title, lecturer) if present
\providecommand{\courseTitle}{Course}
\providecommand{\courseNumber}{000000}
\providecommand{\courseIndicoUrl}{https://indico.global/}
\providecommand{\courseMoodleUrl}{https://moodle.ruhr-uni-bochum.de/}
\providecommand{\lecturer}{Lecturer}
\providecommand{\teachingAssistants}{Teaching Assistants}
\providecommand{\correctors}{Correctors}
\providecommand{\editors}{Editors}
\providecommand{\semester}{Summer/Winter Semester 20XX}

% Title configuration
\title{Exercise sheet \sheetname\ifsolutions\ [Solutions]\fi}
\author{\lecturer}
\date{Due: \@duedate}

% Custom header with logos
\makeatletter
\renewcommand{\@maketitle}{%
    \newpage
    \null
    % Keep first line at same distance from top (was 0.8cm+0.2cm): cancel extra geometry top
    \vspace*{-0.8cm}%
    \vskip 0.2cm%
    \begin{center}%
        % Top section with university info
        \rule{\textwidth}{2pt}\\[0.3em]
        \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcr}
		\textbf{RU Bochum \& TU Dortmund} & \textbf{160435} & \textbf{SoSe 2026} \\
        \end{tabular*}\\[0.5em]

        % Main header section with logos and course info
        \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l@{}c@{}r@{}}
            % Left: agmikhasenko logo
            \begin{minipage}[c]{0.22\textwidth}
                \centering
                \includegraphics[height=1.8cm]{./img/ep1mikhasenko_logo.png}
            \end{minipage} &
            % Center: Course title and instructors
            \begin{minipage}[c]{0.56\textwidth}
                \centering
                {\Large\textcolor[HTML]{17365C}{\textbf{\courseTitle}}}\\[0.2em]
                {\Large\textbf{\@title}}\\[0.3em]
                % {\Large\colorbox[rgb]{0,0.208,0.376}{\textcolor{white}{\textbf{Exercise \sheetname\ifsolutions\ (Solutions)\fi}}}}\\[0.3em]
                \textbf{Prof.~Dr.~Mikhail~Mikhasenko}\\[0.1em]
                \textbf{Assistants:} \teachingAssistants
            \end{minipage} &
            % Right: Course logo
            \begin{minipage}[c]{0.22\textwidth}
                \centering
                \includegraphics[height=2.4cm]{./img/stat-ml-course-logo.png}
            \end{minipage}
        \end{tabular*}\\[0.5em]

        % Bottom section with dates
        \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcr}
            \textbf{\@published} & \href{\courseIndicoUrl}{Indico schedule} & \href{\courseMoodleUrl}{RUB Moodle} \\
        \end{tabular*}\\[0.3em]

        \rule{\textwidth}{2pt}\\[0.5em]
    \end{center}%
    \par
    \vskip 1.5em%
}
\makeatother
