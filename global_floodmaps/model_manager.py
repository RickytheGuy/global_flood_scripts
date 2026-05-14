import os
import multiprocessing as mp

import tqdm

from .domain import Domain

def _run_arc(config: str):
    from arc import Arc
    Arc(config, quiet=True).run()

def _run_c2f(config: str):
    from curve2flood import Curve2Flood_MainFunction
    Curve2Flood_MainFunction(config, quiet=True, flood_vdt_cells=False).run()

class ModelManager:
    def __init__(self, domains: Domain | list[Domain]):
        self.domains = domains if isinstance(domains, list) else [domains]

    def get_arc_tasks(self, overwrite=False) -> list[str]:
        tasks = []
        for domain in self.domains:
            if domain.arc_config and (not os.path.exists(domain.vdt) or overwrite):
                tasks.append(domain.arc_config)
        return tasks
    
    def get_bathy_tasks(self, overwrite=False) -> list[str]:
        tasks = []
        for domain in self.domains:
            if domain.arc_config and domain.burned_dem and (not os.path.exists(domain.flood_flow_file) or overwrite):
                tasks.append(domain.arc_config)
        return tasks

    @profile
    def run(self, overwrite=False, processes: int = 1, pbar: bool = True):
        # Sort domains by their DEM size, largest to smallest, to optimize memory usage
        self.domains.sort(key=lambda d: d.get_priority())

        if processes < 0:
            processes = mp.cpu_count() + processes

        if processes < 1:
            raise ValueError("Number of processes must be at least 1.")
        
        if processes == 1:
            arc_tasks = self.get_arc_tasks(overwrite=overwrite)
            if arc_tasks:
                for config in tqdm.tqdm(arc_tasks, desc="Running ARC", disable=not pbar):
                    _run_arc(config)

            bathy_tasks = self.get_bathy_tasks(overwrite=overwrite)
            if bathy_tasks:
                for config in tqdm.tqdm(bathy_tasks, desc="Running C2F", disable=not pbar):
                    _run_c2f(config)

            return
    
        with mp.Pool(processes) as pool:
            arc_tasks = self.get_arc_tasks(overwrite=overwrite)
            if arc_tasks:
                for _ in tqdm.tqdm(pool.imap_unordered(_run_arc, arc_tasks), desc="Running ARC", disable=not pbar):
                    pass

            bathy_tasks = self.get_bathy_tasks(overwrite=overwrite)
            if bathy_tasks:
                for _ in tqdm.tqdm(pool.imap_unordered(_run_c2f, bathy_tasks), desc="Running C2F", disable=not pbar):
                    pass
            
