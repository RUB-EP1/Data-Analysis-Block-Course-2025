ENV["GKSwstype"]="100"
using Pluto
session=Pluto.ServerSession()
session.options.server.disable_writing_notebook_files=true
out=joinpath(@__DIR__,"exports");mkpath(out)
for file in ["cuts-and-impurity.jl","classification-trees.jl","regression-trees.jl"]
    println("PLUTO RUN ",file)
    notebook=Pluto.SessionActions.open(session,joinpath(@__DIR__,file);run_async=false)
    failed=filter(c->c.errored,notebook.cells)
    for c in failed
        println("CELL ERROR: ",c.code,"\n",c.output.body)
    end
    isempty(failed) || error("Pluto execution failed: $file")
    write(joinpath(out,replace(file,".jl"=>".html")),Pluto.generate_html(notebook))
    println("PLUTO PASS ",file," (",length(notebook.cells)," cells)")
    Pluto.SessionActions.shutdown(session,notebook)
end
