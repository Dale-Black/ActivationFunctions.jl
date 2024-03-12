### A Pluto.jl notebook ###
# v0.19.40

#> [frontmatter]
#> title = "API"

using Markdown
using InteractiveUtils

# ╔═╡ 375ccfb0-cb99-406e-8bd3-6327abf87871
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()

	using PlutoUI, ActivationFunctions
end;

# ╔═╡ e1f2e5e5-ae17-416e-be7b-762938702812
TableOfContents()

# ╔═╡ 2788af30-cca6-47a7-b5f0-32f98f354fcd
md"""
# References
"""

# ╔═╡ efb735c1-a50f-4afe-8ad6-5054a4dbedb6
all_names = [name for name in names(ActivationFunctions)];

# ╔═╡ 55dfaae1-5f95-4102-8d2c-1a8f068a6080
exported_functions = filter(x -> x != :ActivationFunctions, all_names);

# ╔═╡ 64840967-5122-48a1-843a-5604ee298aec
function generate_docs(exported_functions)
    PlutoUI.combine() do Child
        md"""
        $([md" $(Docs.doc(eval(name)))" for name in exported_functions])
        """
    end
end;

# ╔═╡ beefb858-76d3-45a5-8123-89269f73dd9b
generate_docs(exported_functions)

# ╔═╡ Cell order:
# ╟─2788af30-cca6-47a7-b5f0-32f98f354fcd
# ╟─beefb858-76d3-45a5-8123-89269f73dd9b
# ╟─375ccfb0-cb99-406e-8bd3-6327abf87871
# ╟─e1f2e5e5-ae17-416e-be7b-762938702812
# ╟─efb735c1-a50f-4afe-8ad6-5054a4dbedb6
# ╟─55dfaae1-5f95-4102-8d2c-1a8f068a6080
# ╟─64840967-5122-48a1-843a-5604ee298aec
