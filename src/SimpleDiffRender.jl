using Random

import Base: +, -, *, /, <
import LinearAlgebra: ⋅

struct Vec2{T <: Real}
    x::T
    y::T
    Vec2{T}(_x, _y) where T = new{T}(_x, _y)
end

struct Vec3{T <: Real}
    x::T
    y::T
    z ::T
    Vec3{T}(_x, _y, _z) where T = new{T}(_x, _y, _z)
end

Vec2f = Vec2{Float32}

(+)(v0::Vec2f, v1::Vec2f) = Vec2f(v0.x + v1.x, v0.y + v1.y)
(-)(v0::Vec2f, v1::Vec2f) = Vec2f(v0.x - v1.x, v0.y - v1.y)
(*)(n, v1::Vec2f) = Vec2f(n * v1.x, n * v1.y)
(*)(v0::Vec2f, n) = Vec2f(v0.x * n, v0.y * n)
(/)(v0::Vec2f, n) = Vec2f(v0.x / n, v0.y / n)

dot(v0::Vec2f, v1::Vec2f) = v0.x * v1.x + v0.y * v1.y
@inline (⋅)(v0::Vec2f, v1::Vec2f) = dot(v0, v1)
len(v::Vec2f) = √(v ⋅ v)
normal(v::Vec2f) = Vec2f(-v.y, v.x)

Vec3i = Vec3{Int32}
Vec3f = Vec3{Float32}

(+)(v0::Vec3f, v1::Vec3f) = Vec3f(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z)
(-)(v0::Vec3f, v1::Vec3f) = Vec3f(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z)
(-)(v::Vec3f) = Vec3f(-v.x, -v.y, -v.z)
(*)(n, v::Vec3f) = Vec3f(n * v.x, n * v.y, n * v.z)
(*)(v::Vec3f, n) = Vec3f(v.x * n, v.y * n, v.z * n)
(/)(v::Vec3f, n) = Vec3f(v.x / n, v.y / n, v.z / n)
dot(v0::Vec3f, v1::Vec3f) = v0.x * v1.x + v0.y * v1.y + v0.z * v1.z
@inline (⋅)(v0::Vec2f, v1::Vec2f) = dot(v0, v1)

function clamp(v, l, u)
    if v < l
        return l
    elseif v > u
        return u
    else
        return v
    end
end

struct TriangleMesh
    vertices::Vector{Vec2f}
    indices::Vector{Vec3i}
    colors::Vector{Vec3f}
end

struct DTriangleMesh
    vertices::Vector{Vec2f}
    colors::Vector{Vec3f}

    DTriangleMesh(num_verts, num_colors) = new(Vector{Vec2f}(Vec2f(0, 0), num_verts), 
        Vector{Vec3f}(Vec3f(0, 0, 0), num_colors))
end

struct Edge
    v0::Int32
    v1::Int32

    Edge(v0, v1) = new(min(v0, v1), max(v0, v1))
end

(<)(e0::Edge, e1::Edge) = e0.v0 != e1.v0 ? e0.v0 < e1.v0 : e0.v1 < e1.v1

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
    for i = 1:length(mesh.indices)
        index = mesh.indices[i]
        v0 = mesh.vertices[index.x]
        v1 = mesh.vertices[index.y]
        v2 = mesh.vertices[index.z]
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

function render(w, h, mesh, samples, rng)
    sqrt_s = Int32(√samples)
    spp = sqrt_s * sqrt_s
    img = Vector{Vec3f}(undef, w * h)
    fill!(img, Vec3f(0, 0, 0))
    for y = 1:h
        for x = 1:w
            for dy = 1:sqrt_s
                for dx = 1:sqrt_s
                    xoff = (dx + rand(rng)) / sqrt_s
                    yoff = (dy + rand(rng)) / sqrt_s
                    screen_pos = Vec2f(x + xoff, y + yoff)
                    intersection = raytrace(mesh, screen_pos)
                    img[(y - 1) * w + x] += intersection.shade / spp
                end
            end
        end
    end
    return reshape(img, (w, h))
end

tonemap(rgb) = Int32(round(clamp(rgb, 0.f0, 1.f0) ^ (1.f0 / 2.2f0) * 255.99f0))

function save(img, name; flip = false)
    open(name, "w") do io
        println(io, "P3\n$(size(img)[1]) $(size(img)[2]) 255")
        for i = 1:length(img)
            color = flip ? -img[i] : img[i]
            println(io , tonemap(color.x), " ", tonemap(color.y), " ", tonemap(color.z))
        end
    end
end

function main()
    width, height = 256, 256
    mesh = TriangleMesh(
        # vertices
        [Vec2f(50.f0, 25.f0), Vec2f(200.f0, 200.f0), Vec2f(15.f0, 150.f0), 
        Vec2f(200.f0, 15.f0), Vec2f(150.f0, 250.f0), Vec2f(50.f0, 100.f0)],
        #indices
        [Vec3i(1, 2, 3),
        Vec3i(4, 5, 6)],
        # colors
        [Vec3f(0.3f0, 0.5f0, 0.3f0),
        Vec3f(0.3f0, 0.3f0, 0.5f0)])
    rng = MersenneTwister(1234)
    img = render(width, height, mesh, 4, rng)
    #save(img, "render.ppm")
end