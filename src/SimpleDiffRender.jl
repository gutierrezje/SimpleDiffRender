using Random
using StaticArrays
using LinearAlgebra

import Base: isless, isequal

Vec2f = SVector{2, Float32}

len(v::Vec2f) = √(v ⋅ v)
normal(v::Vec2f) = Vec2f(-v[2], v[1])

Vec3f = SVector{3, Float32}
Vec3i = SVector{3, Int32}

# Some utilities
function clamp(v, l, u)
    if v < l
        return l
    elseif v > u
        return u
    else
        return v
    end
end

tonemap(rgb) = Int32(round(clamp(rgb, 0.f0, 1.f0) ^ (1.f0 / 2.2f0) * 255.99f0))

struct TriangleMesh
    vertices::Vector{Vec2f}
    indices::Vector{Vec3i}
    colors::Vector{Vec3f}
end

struct DTriangleMesh
    vertices::Vector{Vec2f}
    colors::Vector{Vec3f}

    function DTriangleMesh(num_verts, num_colors)
        vertices = Vector{Vec2f}(undef, num_verts)
        fill!(vertices, Vec2f(0, 0))
        colors = Vector{Vec3f}(undef, num_colors)
        fill!(colors, Vec3f(0, 0, 0))
        return new(vertices, colors)
    end
end

struct Edge
    v0::Int32
    v1::Int32

    Edge(v0, v1) = new(min(v0, v1), max(v0, v1))
end

isless(e0::Edge, e1::Edge) = e0.v0 != e1.v0 ? e0.v0 < e1.v0 : e0.v1 < e1.v1
isequal(e0::Edge, e1::Edge) = (e0.v0 == e1.v0) && (e0.v1 == e1.v1)

struct Sampler
    pmf::Vector{Float32}
    cdf::Vector{Float32}
end

struct Intersection
    shade::Vec3f
    index::Int32

    Intersection() = new(Vec3f(0, 0, 0), -1)
    Intersection(s, i) = new(s, i)
end

function raytrace(mesh, screen_pos)
    for i in 1:length(mesh.indices)
        index = mesh.indices[i]
        v0 = mesh.vertices[index[1]]
        v1 = mesh.vertices[index[2]]
        v2 = mesh.vertices[index[3]]
        n01 = normal(v1 - v0)
        n12 = normal(v2 - v1)
        n20 = normal(v0 - v2)
        side01 = (screen_pos - v0) ⋅ n01 > 0
        side12 = (screen_pos - v1) ⋅ n12 > 0
        side20 = (screen_pos - v2) ⋅ n20 > 0
        if (side01 && side12 && side20) || (!side01 && !side12 && !side20)
            return Intersection(mesh.colors[i], i)
        end
    end
    return Intersection()
end

function render!(img, w, h, mesh, samples, rng)
    sqrt_num_samples = Int32(√samples)
    spp = sqrt_num_samples * sqrt_num_samples
    for y in 1:h
        for x in 1:w
            for dy in 1:sqrt_num_samples
                for dx in 1:sqrt_num_samples
                    xoff = (dx + rand(rng)) / sqrt_num_samples
                    yoff = (dy + rand(rng)) / sqrt_num_samples
                    screen_pos = Vec2f(x + xoff, y + yoff)
                    intersection = raytrace(mesh, screen_pos)
                    img[(y - 1) * w + x] += intersection.shade / spp
                end
            end
        end
    end
end

function save(img, w, h, name; flip = false)
    open(name, "w") do io
        println(io, "P3\n$(w) $(h) 255")
        for i = 1:length(img)
            color = flip ? -img[i] : img[i]
            println(io , tonemap(color[1]), " ", tonemap(color[2]), " ", tonemap(color[3]))
        end
    end
end

function build_edge_sampler(mesh, edges)
    pmf = Vector{Float32}()
    sizehint!(pmf, length(edges))
    cdf = Vector{Float32}()
    sizehint!(cdf, length(edges) + 1)
    push!(cdf, 0)
    for edge in edges
        v0 = mesh.vertices[edge.v0]
        v1 = mesh.vertices[edge.v1]
        push!(pmf, len(v1 - v0))
        push!(cdf, pmf[end] + cdf[end])
    end
    length_sum = cdf[end]
    for p in pmf
        p /= length_sum
    end
    for c in cdf
        c /= length_sum
    end
    return Sampler(pmf, cdf)
end

function sample(sampler, u)
    first_gt = searchsortedfirst(sampler.cdf, u)
    return Int32(clamp(first_gt, 1, length(sampler.cdf) - 1))
end

function collect_edges(mesh)
    edges = Set{Edge}()
    for index in mesh.indices
        push!(edges, Edge(index[1], index[2]))
        push!(edges, Edge(index[2], index[3]))
        push!(edges, Edge(index[3], index[1]))
    end

    return sort!(collect(edges))
end

function compute_interior_derivatives(d_colors, mesh, spp, w, h, adjoint, rng)
    sqrt_num_samples = Int32(sqrt(spp))
    spp = sqrt_num_samples * sqrt_num_samples
    for y = 1:h
        for x = 1:w
            for dy = 1:sqrt_num_samples
                for dx = 1:sqrt_num_samples
                    xoff = (dx + rand(rng)) / sqrt_num_samples
                    yoff = (dy + rand(rng)) / sqrt_num_samples
                    screen_pos = Vec2f(x + xoff, y + yoff)
                    hit = raytrace(mesh, screen_pos)
                    if hit.index != -1
                        d_colors[hit.index] += adjoint[(y - 1) * w + x] / spp
                    end
                end
            end
        end
    end
end

function compute_edge_derivatives(mesh, edges, sampler, adjoint, num_edge_samples, w, h, rng, screen_dx, screen_dy, d_vertices)
    ϵ = 1f-3
    for i = 1:num_edge_samples
        edge_id = sample(sampler, rand(rng))
        edge = edges[edge_id]
        pmf = sampler.pmf[edge_id]
        v0 = mesh.vertices[edge.v0]
        v1 = mesh.vertices[edge.v1]
        t = rand(rng)
        p = v0 + t * (v1 - v0)
        xi = Int32(floor(p[1]))
        yi = Int32(floor(p[2]))
        if xi < 0 || yi < 0 || xi > w || yi > h
            continue
        end
        n = normal((v1 - v0) / len(v1 - v0))
        color_in = raytrace(mesh, p - n * ϵ).shade
        color_out = raytrace(mesh, p + n * ϵ).shade
        
        pdf = pmf / len(v1 - v0)
        weight = 1f0 / (pdf * num_edge_samples)
        adj = (color_in - color_out) ⋅ (adjoint[(yi - 1) * w + xi])

        d_v0 = Vec2f((1 - t) * n[1], (1 - t) * n[2]) * adj * weight
        d_v1 = Vec2f(t * n[1], t * n[2]) * adj * weight

        dx = -n[1] * (color_in - color_out) * weight
        dy = -n[2] * (color_in - color_out) * weight
        
        screen_dx[(yi - 1) * w + xi] += dx
        screen_dy[(yi - 1) * w + xi] += dy
        d_vertices[edge.v0] += d_v0
        d_vertices[edge.v1] += d_v1
    end
end

function d_render!(screen_dx, screen_dy, adjoint, interior_spp, w, h, mesh, rng, d_mesh)
    edge_samples_total = w * h
    compute_interior_derivatives(d_mesh.colors, mesh, interior_spp, w, h, adjoint, rng)
    edges = collect_edges(mesh)
    edge_sampler = build_edge_sampler(mesh, edges)
    compute_edge_derivatives(mesh, edges, edge_sampler, adjoint, edge_samples_total, w, h, rng, screen_dx, screen_dy, d_mesh.vertices)
end

function main()
    w, h = 256, 256
    vertices = [Vec2f(50., 25.), Vec2f(200., 200.), Vec2f(15., 150.), 
        Vec2f(200., 15.), Vec2f(150., 250.), Vec2f(50., 100.)]
    indices = [Vec3i(1, 2, 3),
        Vec3i(4, 5, 6)]
    colors = [Vec3f(0.3, 0.5, 0.3),
        Vec3f(0.3, 0.3, 0.5)]
    mesh = TriangleMesh(vertices, indices, colors)

    rng = MersenneTwister(1234)
    img = zeros(Vec3f, w * h)
    render!(img, w, h, mesh, 4, rng)
    save(img, w, h, "render.ppm")

    adjoint = fill(Vec3f(1, 1, 1), w * h)
    dx = zeros(Vec3f, w * h)
    dy = zeros(Vec3f, w * h)
    d_mesh = DTriangleMesh(length(mesh.vertices), length(mesh.colors))
    d_render!(dx, dy, adjoint, 4, w, h, mesh, rng, d_mesh)
    save(dx, w, h, "dx_pos.ppm")
    save(dx, w, h, "dx_neg.ppm", flip = true)
    save(dy, w, h, "dy_pos.ppm")
    save(dy, w, h, "dy_neg.ppm", flip = true)
end

main()
