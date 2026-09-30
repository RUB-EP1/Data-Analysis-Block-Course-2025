# Pluto server for the live Julia cells embedded in a deck.
#
#   julia scripts/serve_pluto.jl Lecture_3A      # or: make start-julia-server DECK=Lecture_3A
#
# The notebooks come from lectures/<DECK>/pluto.toml, each with a fixed notebook id,
# so the slide URLs stay valid across restarts. A slide embeds Pluto's isolated-cell
# view, which shows only the chosen cells with their @bind sliders live:
#
#   http://localhost:1243/edit?id=<id>&isolated_cell_id=<cell>&isolated_cell_id=<cell>
#
# The secret is off because the slide iframe cannot know it; Pluto listens on
# 127.0.0.1 only. Notebook files are never written.
using Pluto, TOML

const PORT = parse(Int, get(ENV, "PLUTO_PORT", "1243"))  # also in the slide URLs

deck = isempty(ARGS) ? error("usage: julia scripts/serve_pluto.jl <DECK>") : ARGS[1]
deck_dir = joinpath(@__DIR__, "..", "lectures", deck)
registry = joinpath(deck_dir, "pluto.toml")
isfile(registry) || error("$deck has no Pluto notebooks ($registry not found)")

session = Pluto.ServerSession(; options=Pluto.Configuration.from_flat_kwargs(;
    port=PORT,
    launch_browser=false,
    require_secret_for_access=false,
    require_secret_for_open_links=false,
    disable_writing_notebook_files=true,
))
for nb in TOML.parsefile(registry)["notebook"]
    Pluto.SessionActions.open(session, joinpath(deck_dir, nb["path"]);
        notebook_id=Base.UUID(nb["id"]), run_async=true)
    @info "$deck: $(nb["path"])" url = "http://localhost:$PORT/edit?id=$(nb["id"])"
end

Pluto.run(session)
