# Included inside MainFigureAssets; presentation-only catalogue.
function gallery_pair(io,heading,left,right)
    isempty(heading) || println(io,"<h3>$(xml(heading))</h3>")
    println(io,"<div class=\"pair\">")
    for (prefix,file) in (("dense",left),("sparse",right))
        caption=prefix=="dense" ? "Dense · Lila" : "Sparse · Magenta"
        println(io,"<figure class=\"$prefix\"><div class=\"side-label\">$caption</div><a href=\"$file\"><img loading=\"lazy\" src=\"$file\" alt=\"$file\"></a><figcaption><a href=\"$file\" download>SVG herunterladen</a><span>$(xml(basename(file)))</span></figcaption></figure>")
    end
    println(io,"</div>")
end

function write_gallery(path,data)
    open(path,"w") do io
        println(io,read(joinpath(@__DIR__,"gallery_header.html"),String))
        println(io,"<p>Ein Punkt repräsentiert einen Sensorort mit allen drei Messkanälen. Seine Farbe zeigt ausschließlich die Temperatur. Dense: 384 Orte. Sparse: $(count(data.mask[1,:,:])) Orte mit der unveränderten Varying-IC-GO-GC-Maske.</p>")
        gallery_pair(io,"Sensorpunkte ohne Windows","dense_temperature.svg","sparse_temperature.svg")
        println(io,"<h2>Window-Variationen</h2><p>Gleiche Agentnummer, gleiche Farbe über alle Auswahlen. Alle zwölf Aktuatoren bleiben sichtbar. Geteilte Randfenster sind mit ihrer Agentnummer markiert.</p>")
        println(io,"""
        <div class="controls">
          <label>Darstellung <select id="style"><option value="frames">Rahmen und Klammern</option><option value="brackets">Nur Klammern</option><option value="all">Beide Darstellungen</option></select></label>
          <label>Auswahl <select id="count"><option value="all">Alle Auswahlen</option><option value="1">1 Agent</option><option value="2">2 Agenten</option><option value="3">3 Agenten</option><option value="4">4 Agenten</option><option value="6">6 Agenten</option><option value="12">Alle 12 Agenten</option></select></label>
          <span id="shown" aria-live="polite"></span>
        </div>
        """)
        for v in WINDOW_VARIANTS, style in ("frames","brackets")
            style_name=style=="frames" ? "Rahmen und Klammern" : "Nur Klammern"
            println(io,"<section class=\"variant\" data-style=\"$style\" data-count=\"$(length(v.agents))\"><h3>$(xml(v.title)) <span>$(join(v.agents, ", ")) · $style_name</span></h3><p>$(xml(v.note))</p>")
            gallery_pair(io,"","windows/dense_$(v.id)_$style.svg","windows/sparse_$(v.id)_$style.svg")
            println(io,"</section>")
        end
        println(io,"<h2>Weitere Bausteine</h2><details><summary>Lokale Windows, Controller, Aktuatoren und Legenden</summary>")
        for (heading,left,right) in (
            ("Lokale Beobachtungen der Standardauswahl", "dense_local_windows.svg", "sparse_local_windows.svg"),
            ("Alle zwölf lokalen Windows separat", "dense_all_local_windows.svg", "sparse_all_local_windows.svg"),
            ("Controller-Bausteine", "dense_controller.svg", "sparse_controller.svg"),
            ("Zwölf Aktuatoren", "dense_actuators_12.svg", "sparse_actuators_12.svg"))
            gallery_pair(io,heading,left,right)
        end
        for file in ("two_plume_temperature_field.svg","temperature_legend.svg","agent_palettes.svg","distillation_arrow.svg")
            println(io,"<figure class=\"single\"><a href=\"$file\"><img loading=\"lazy\" src=\"$file\" alt=\"$file\"></a><figcaption><a href=\"$file\" download>$file</a></figcaption></figure>")
        end
        archive=joinpath(@__DIR__,"iterations","01","assets","index.html")
        archive_link=isfile(archive) ? " · <a href=\"$(xml(replace(relpath(archive,dirname(path)), '\\'=>'/')))\">Iteration 1</a>" : ""
        println(io,"</details><p><a href=\"provenance.json\">Datenherkunft und Window-Zuordnungen</a>$archive_link</p>")
        println(io,"""
        </main><script>
        function filterVariants() {
          const style=document.getElementById('style').value;
          const count=document.getElementById('count').value;
          let shown=0;
          document.querySelectorAll('.variant').forEach(section=>{
            const visible=(style==='all'||section.dataset.style===style)&&(count==='all'||section.dataset.count===count);
            section.hidden=!visible;
            shown+=visible?1:0;
          });
          document.getElementById('shown').textContent=shown+' Variantenpaare';
        }
        document.querySelectorAll('select').forEach(select=>select.addEventListener('change',filterVariants));
        filterVariants();
        </script></body></html>
        """)
    end
end
