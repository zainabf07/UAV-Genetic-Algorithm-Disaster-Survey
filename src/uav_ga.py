"""
UAV Genetic Algorithm Strategy Engine
Artificial Intelligence — BSCS 5 — Assignment 2

A reproducible educational simulation for multi-UAV disaster-zone surveying.
This is a simplified grid-world model, not flight-certified software.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

GRID_SIZE = 50
N_DRONES = 3
N_WAYPOINTS = 6
GENES_PER_WAYPOINT = 4  # x, y, normalized speed, sensor power
CHROMOSOME_SHAPE = (N_DRONES, N_WAYPOINTS, GENES_PER_WAYPOINT)

@dataclass
class Config:
    population_size: int = 60
    generations: int = 100
    mutation_rate: float = 0.12
    tournament_size: int = 3
    elite_count: int = 2
    seed: int = 42

class DisasterGrid:
    """50x50 grid: 0=open, 1=no-fly obstacle, continuous hazard and priority maps."""
    def __init__(self, size=GRID_SIZE, seed=7):
        self.size = size
        rng = np.random.default_rng(seed)
        self.blocked = np.zeros((size, size), dtype=bool)
        # Rectangular restricted regions, leaving a feasible route network.
        rectangles = [(8, 8, 13, 18), (22, 4, 27, 15), (34, 20, 40, 30),
                      (10, 32, 17, 39), (27, 36, 32, 44)]
        for x1, y1, x2, y2 in rectangles:
            self.blocked[y1:y2+1, x1:x2+1] = True
        # Smooth-ish synthetic environmental hazard field.
        yy, xx = np.mgrid[0:size, 0:size]
        self.hazard = np.zeros((size, size), dtype=float)
        for cx, cy, amp, sig in [(18, 25, .95, 7), (38, 10, .8, 6), (30, 33, .7, 8)]:
            self.hazard += amp*np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*sig**2))
        self.hazard = np.clip(self.hazard + rng.uniform(0, .12, (size,size)), 0, 1)
        self.hazard[self.blocked] = 1.0
        self.priority = np.zeros((size, size), dtype=float)
        # High-priority disaster zones / targets.
        for cx, cy, amp, sig in [(6, 40, 1.0, 5), (20, 20, .9, 6),
                                 (43, 38, 1.0, 5), (41, 5, .75, 4)]:
            self.priority += amp*np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*sig**2))
        self.priority = np.clip(self.priority, 0, 1)
        self.priority[self.blocked] = 0
        self.starts = np.array([[2, 2], [2, 47], [47, 2]], dtype=float)

    def is_blocked(self, x, y):
        xi, yi = int(round(np.clip(x, 0, self.size-1))), int(round(np.clip(y, 0, self.size-1)))
        return bool(self.blocked[yi, xi])

class UAVGeneticAlgorithm:
    def __init__(self, grid=None, config=None):
        self.grid = grid or DisasterGrid()
        self.config = config or Config()
        self.rng = np.random.default_rng(self.config.seed)
        self.history = {"best": [], "average": []}
        self.best = None
        self.best_metrics = None

    def random_chromosome(self):
        # Genes per waypoint: x,y in grid; speed in [0.5,1.5]; sensor power in [0,1].
        c = np.empty(CHROMOSOME_SHAPE, dtype=float)
        c[:,:,0:2] = self.rng.uniform(0, self.grid.size-1, (N_DRONES,N_WAYPOINTS,2))
        c[:,:,2] = self.rng.uniform(.5, 1.5, (N_DRONES,N_WAYPOINTS))
        c[:,:,3] = self.rng.uniform(.2, 1.0, (N_DRONES,N_WAYPOINTS))
        return self.repair(c)

    def repair(self, chromosome):
        c = np.array(chromosome, dtype=float, copy=True).reshape(CHROMOSOME_SHAPE)
        c[:,:,0:2] = np.clip(c[:,:,0:2], 0, self.grid.size-1)
        c[:,:,2] = np.clip(c[:,:,2], .5, 1.5)
        c[:,:,3] = np.clip(c[:,:,3], 0, 1)
        # Repair waypoint coordinates that land in a no-fly cell by local search.
        for d in range(N_DRONES):
            for w in range(N_WAYPOINTS):
                x, y = c[d,w,0:2]
                if self.grid.is_blocked(x,y):
                    found = False
                    for radius in range(1, 8):
                        candidates = [(x+dx,y+dy) for dx in range(-radius,radius+1)
                                      for dy in range(-radius,radius+1)
                                      if abs(dx)==radius or abs(dy)==radius]
                        self.rng.shuffle(candidates)
                        for nx,ny in candidates:
                            nx,ny = np.clip([nx,ny],0,self.grid.size-1)
                            if not self.grid.is_blocked(nx,ny):
                                c[d,w,0:2] = [nx,ny]; found=True; break
                        if found: break
        return c

    def route_points(self, chromosome, drone):
        start = self.grid.starts[drone]
        goals = chromosome[drone,:,0:2]
        nodes = np.vstack([start, goals])
        route=[]
        for i in range(len(nodes)-1):
            a,b=nodes[i],nodes[i+1]
            steps=max(2,int(np.ceil(np.linalg.norm(b-a)/3.0)))
            seg=np.linspace(a,b,steps+1)
            route.extend(seg[1:])
        return np.asarray(route)

    def evaluate(self, chromosome):
        c = np.asarray(chromosome, dtype=float).reshape(CHROMOSOME_SHAPE)
        covered = np.zeros((self.grid.size,self.grid.size), dtype=bool)
        risk_sum=0.0; distance=0.0; blocked_hits=0; time_sum=0.0; energy_sum=0.0
        detected_weight=0.0; samples=0
        for d in range(N_DRONES):
            nodes=np.vstack([self.grid.starts[d], c[d,:,0:2]])
            for w in range(N_WAYPOINTS):
                a,b=nodes[w],nodes[w+1]
                distance += float(np.linalg.norm(b-a))
                # Sample each segment at a coarse but consistent resolution.
                steps=max(2,int(np.ceil(np.linalg.norm(b-a)/5.0)))
                seg=np.linspace(a,b,steps+1)
                speed=float(c[d,w,2]); power=float(c[d,w,3])
                for x,y in seg[1:]:
                    xi=int(np.clip(round(x),0,49)); yi=int(np.clip(round(y),0,49))
                    blocked_hits += int(self.grid.blocked[yi,xi])
                    risk_sum += float(self.grid.hazard[yi,xi]); samples += 1
                    radius=1+int(round(2*power))
                    y0,y1=max(0,yi-radius),min(49,yi+radius)
                    x0,x1=max(0,xi-radius),min(49,xi+radius)
                    yy,xx=np.ogrid[y0:y1+1,x0:x1+1]
                    mask=(xx-xi)**2+(yy-yi)**2 <= radius**2
                    covered[y0:y1+1,x0:x1+1] |= mask
                    detected_weight += float(self.grid.priority[yi,xi])*power
                time_sum += float(np.linalg.norm(b-a))/max(.5,speed)
                energy_sum += float(np.linalg.norm(b-a))*(.55+.25*speed**2)+2.5*power**2
        coverage=float((self.grid.priority*covered).sum()/(self.grid.priority.sum()+1e-9))
        detection=float(np.clip(detected_weight/(max(1,samples)*.35),0,1))
        mean_risk=risk_sum/max(1,samples)
        energy_norm=min(1.0,energy_sum/(N_DRONES*GRID_SIZE*2.5))
        time_norm=min(1.0,time_sum/(N_DRONES*GRID_SIZE*1.8))
        range_excess=max(0.0,distance-N_DRONES*GRID_SIZE*1.35)/(N_DRONES*GRID_SIZE)
        violation=3.0*blocked_hits+5.0*range_excess
        fitness=(.48*coverage+.22*detection-.13*energy_norm-.07*time_norm-.10*mean_risk-.02*violation)
        return float(fitness), {"fitness":float(fitness),"coverage":coverage,"detection":detection,
          "energy":float(energy_sum),"energy_norm":float(energy_norm),"time":float(time_sum),
          "time_norm":float(time_norm),"risk":float(mean_risk),"blocked_hits":int(blocked_hits),
          "distance":float(distance),"range_excess":float(range_excess)}

    def tournament(self, population, scores):
        ids=self.rng.integers(0,len(population),size=self.config.tournament_size)
        winner=ids[np.argmax(scores[ids])]
        return population[winner].copy()

    def crossover(self, p1, p2):
        # Segment-swapping crossover: select one drone/waypoint block boundary.
        a,b=p1.copy(),p2.copy()
        cut_d=int(self.rng.integers(0,N_DRONES))
        cut_w=int(self.rng.integers(0,N_WAYPOINTS))
        child=a.copy()
        child[cut_d,cut_w:,:]=b[cut_d,cut_w:,:]
        if self.rng.random()<.5:
            child[cut_d+1:,:,:]=b[cut_d+1:,:,:]
        return self.repair(child)

    def mutate(self, chromosome):
        c=chromosome.copy()
        for d in range(N_DRONES):
            for w in range(N_WAYPOINTS):
                if self.rng.random()<self.config.mutation_rate:
                    c[d,w,0:2] += self.rng.normal(0,5.0,2)
                if self.rng.random()<self.config.mutation_rate:
                    c[d,w,2] += self.rng.normal(0,.18)
                if self.rng.random()<self.config.mutation_rate:
                    c[d,w,3] += self.rng.normal(0,.15)
        return self.repair(c)

    def run(self):
        pop=np.array([self.random_chromosome() for _ in range(self.config.population_size)])
        for _ in range(self.config.generations):
            evaluated=[self.evaluate(c) for c in pop]
            scores=np.array([e[0] for e in evaluated])
            order=np.argsort(scores)[::-1]
            self.history["best"].append(float(scores[order[0]]))
            self.history["average"].append(float(scores.mean()))
            if self.best is None or scores[order[0]] > self.evaluate(self.best)[0]:
                self.best=pop[order[0]].copy()
                self.best_metrics=evaluated[order[0]][1]
            new=[pop[i].copy() for i in order[:self.config.elite_count]]
            while len(new)<self.config.population_size:
                p1=self.tournament(pop,scores); p2=self.tournament(pop,scores)
                child=self.crossover(p1,p2)
                child=self.mutate(child)
                new.append(child)
            pop=np.asarray(new)
        # Keep final generation's best if it beats previous best.
        evaluated=[self.evaluate(c) for c in pop]
        idx=int(np.argmax([e[0] for e in evaluated]))
        if evaluated[idx][0] > self.evaluate(self.best)[0]:
            self.best=pop[idx].copy(); self.best_metrics=evaluated[idx][1]
        return self.best, self.best_metrics

def save_outputs(out_dir="results", config=None):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    cfg=config or Config()
    grid=DisasterGrid()
    ga=UAVGeneticAlgorithm(grid,cfg)
    best,metrics=ga.run()
    np.save(out/"best_chromosome.npy",best)
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2))
    (out/"history.json").write_text(json.dumps(ga.history,indent=2))
    # Convergence plot.
    plt.figure(figsize=(8,4.5))
    plt.plot(ga.history["best"],label="Best fitness")
    plt.plot(ga.history["average"],label="Average fitness")
    plt.xlabel("Generation"); plt.ylabel("Fitness (higher is better)")
    plt.title("GA Fitness Convergence"); plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig(out/"fitness_convergence.png",dpi=180); plt.close()
    # Flight paths and priority/coverage map.
    fig,ax=plt.subplots(figsize=(7,6))
    ax.imshow(grid.priority,origin="lower",cmap="YlOrRd",alpha=.8)
    ax.imshow(grid.blocked,origin="lower",cmap="gray_r",alpha=.8, vmin=0,vmax=1)
    colors=["tab:blue","tab:green","tab:purple"]
    for d in range(N_DRONES):
        route=ga.route_points(best,d)
        ax.plot(route[:,0],route[:,1],color=colors[d],lw=1.6,label=f"UAV {d+1}")
        ax.scatter([grid.starts[d,0]],[grid.starts[d,1]],marker="s",s=45,color=colors[d])
        ax.scatter(best[d,:,0],best[d,:,1],marker="o",s=18,color=colors[d])
    ax.set_xlim(0,49); ax.set_ylim(0,49); ax.set_xlabel("X cell"); ax.set_ylabel("Y cell")
    ax.set_title("Evolved UAV Routes over Priority Zones and No-Fly Areas")
    ax.legend(); fig.tight_layout(); fig.savefig(out/"uav_routes.png",dpi=180); plt.close(fig)
    # Coverage heatmap.
    coverage=np.zeros((grid.size,grid.size),dtype=float)
    for d in range(N_DRONES):
        route=ga.route_points(best,d)
        for x,y in route:
            xi,yi=int(round(np.clip(x,0,49))),int(round(np.clip(y,0,49)))
            power=float(best[d,min(int(len(route)//N_WAYPOINTS),N_WAYPOINTS-1),3])
            r=1+int(round(2*power))
            y0,y1=max(0,yi-r),min(49,yi+r); x0,x1=max(0,xi-r),min(49,xi+r)
            yy,xx=np.ogrid[y0:y1+1,x0:x1+1]
            coverage[y0:y1+1,x0:x1+1] += (((xx-xi)**2+(yy-yi)**2<=r*r)*1.0)
    fig,ax=plt.subplots(figsize=(7,5.5))
    im=ax.imshow(coverage,origin="lower",cmap="viridis")
    ax.imshow(grid.blocked,origin="lower",cmap="gray_r",alpha=.45,vmin=0,vmax=1)
    fig.colorbar(im,ax=ax,label="Sensor footprint visits")
    ax.set_xlabel("X cell"); ax.set_ylabel("Y cell"); ax.set_title("Sensor Coverage Heatmap")
    fig.tight_layout(); fig.savefig(out/"coverage_heatmap.png",dpi=180); plt.close(fig)
    return metrics,ga.history

if __name__=="__main__":
    m,h=save_outputs()
    print(json.dumps(m,indent=2))
