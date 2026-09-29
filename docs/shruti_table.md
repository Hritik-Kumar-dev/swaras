# The 22-shruti and 12-swara tuning tables

All tuning data lives in [`src/swaras/config/shruti_table.json`](../src/swaras/config/shruti_table.json)
so the musicology is data, not code. This file documents the numbers and the
assumptions behind the mapping.

## The 22 shruti table

The default is a just-intonation scale, one entry per shruti, at 1-cent
resolution:

```
0, 90, 112, 182, 204, 294, 316, 386, 408, 498, 519,
590, 612, 702, 792, 814, 884, 906, 996, 1018, 1088, 1110
```

## Mapping shruti to swara

Each shruti belongs to one of the 12 swara positions. The mapping is not
arbitrary: the 22 shrutis fall into 11 pairs that are 22 to 23 cents wide,
plus two singletons, and each pair sits inside one swara position.

| Shrutis (cents) | Width | Swara | 12-swara nominal (mean) |
| --- | --- | --- | --- |
| 0 | – | Sa | 0 |
| 90, 112 | 22 | Re komal | 101 |
| 182, 204 | 22 | Re | 193 |
| 294, 316 | 22 | Ga komal | 305 |
| 386, 408 | 22 | Ga | 397 |
| 498, 519 | 21 | Ma | 508.5 |
| 590, 612 | 22 | Ma tivra | 601 |
| 702 | – | Pa | 702 |
| 792, 814 | 22 | Dha komal | 803 |
| 884, 906 | 22 | Dha | 895 |
| 996, 1018 | 22 | Ni komal | 1007 |
| 1088, 1110 | 22 | Ni | 1099 |

### Assumptions

1. **Sa and Pa are singletons.** The table gives 22 shrutis over 12 swara
   positions. If all 12 had pairs that would need 24 entries; 22 is two short.
   Sa (0) and Pa (702) are each a single shruti, and the other 10 positions
   form pairs. This is consistent with the gap structure of the data: 0 to 90
   is 90 cents, 702 to 792 is 90 cents, and both Pa and Sa sit a fifth and an
   octave from the octave boundaries rather than inside a 22-cent pair.

2. **The 12-swara nominal is the pair mean.** For Level A mapping
   (`swar.py`) each swara position is placed at the midpoint of its pair. This
   is the only choice that treats a komal swara and its shuddha partner
   symmetrically; the alternative, using the lower shruti as canonical, would
   bias every swara flat by about 11 cents.

3. **Ma tivra is a distinct swara position,** not a sharpened Pa. The task
   specifies the 12 positions as Sa, komal Re, Re, komal Ga, Ga, Ma, tivra
   Ma, Pa, komal Dha, Dha, komal Ni, Ni, which places tivra Ma between Ma and
   Pa. The shrutis at 590 and 612 sit between Ma (498/519) and Pa (702), so
   they map there. Note that in the 22-shruti framework Pa and Ma tivra can
   also be read as Pa komal and Pa shuddha; the table follows the task's
   12-position convention, and the choice only affects which label is printed
   for a pitch near 600 cents, never the pitch itself.

4. **Tivra Ma is rarer than the other altered swaras,** so in practice a
   performance near 600 cents is more often intended as a slightly sharp Pa
   than a genuine tivra Ma. This is a labelling ambiguity inherent to the
   tuning, and the reason the shruti stage reports a pitch and a deviation
   rather than a single hard name.

## Just intonation versus equal temperament

Equal temperament would place the 12 positions at 0, 100, 200, … 1100 cents.
The just-intonation values above differ by up to 17 cents (Ni sits at 1099
instead of 1100, Re at 193 instead of 200). Two consequences:

- A singer trained in just intonation will be measured as slightly flat or
  sharp against this table in a way they are not against equal temperament.
- The shruti stage is the honest interface: it reports the nearest shruti plus
  the deviation in cents, so a performer sitting 8 cents above the nominal Pa
  is described as `Pa +8c`, not forced into a wrong label.

## Editing the table

Copy the JSON, change `cents` values or add/remove positions, and point the
pipeline at it:

```python
from swaras.config import PipelineConfig
config = PipelineConfig(shruti_table=Path("my_table.json"))
```

The loader validates on load: 12 swara positions, 22 shrutis, strictly
increasing cents, and every shruti assigned to a known swara.
