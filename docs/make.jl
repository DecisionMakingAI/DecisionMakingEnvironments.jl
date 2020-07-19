using DecisionMakingEnvironments
using Documenter

makedocs(;
    modules=[DecisionMakingEnvironments],
    authors="Scott Jordan",
    repo="https://github.com/DecisionMakingAI/DecisionMakingEnvironments.jl/blob/{commit}{path}#L{line}",
    sitename="DecisionMakingEnvironments.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://DecisionMakingAI.github.io/DecisionMakingEnvironments.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/DecisionMakingAI/DecisionMakingEnvironments.jl",
)
