# Adjudication Worklist

234 field-level disagreements needing review.

## dev-0001
- **rfi_type**: annotator_b='document_discrepancy'; gold='design_clarification'
- **question_summary**: annotator_b='Curtain wall panel head/jamb detail at Gridline D/7 conflicts between Detail 5/A501 and Sheet A-301; clarification needed to proceed with fabrication.'; gold='Conflicting head/jamb detail for the curtain wall panel at Gridline D/7 between Detail 5/A501 and Sheet A-301.'
- **deadline_text**: annotator_b='none stated'; gold='No specific response date is requested.'
- **routing_rationale**: annotator_b='Architectural discrepancy between drawing details requires the Architect of Record to reconcile conflicting curtain wall specifications and issue clarification.'; gold='Default discipline routing: the Architect of Record interprets and clarifies the Contract Documents for this Architectural-discipline design clarification.'

## dev-0002
- **question_summary**: annotator_b='Request to substitute a fan-powered VAV box for the specified single-duct VAV terminal unit in Section 23 09 00, cited as cost-neutral with no schedule impact.'; gold='Request to substitute a fan-powered alternate VAV box in lieu of the specified single-duct VAV terminal unit, specified in Section 23 09 00.'
- **referenced_documents**: annotator_b=['Section 23 09 00']; gold=[]
- **proposed_solution**: annotator_b='Fan-powered alternate VAV box in lieu of single-duct VAV terminal unit'; gold='—'
- **routing_rationale**: annotator_b='Mechanical Engineer must evaluate the technical equivalency, performance implications, and code compliance of the proposed fan-powered VAV box substitution against the original specification.'; gold='Default discipline routing: the Mechanical Engineer evaluates substitution requests with no cost impact.'

## dev-0003
- **rfi_type**: annotator_b='document_discrepancy'; gold='design_clarification'
- **secondary_disciplines**: annotator_b=['Mechanical']; gold=[]
- **question_summary**: annotator_b='Sprinkler branch line detailing in Fire Alarm Riser Diagram FA-501 at Mechanical Room 3B conflicts with Fire Protection Sheet FP-101; clarification needed before shop drawing release.'; gold='Inconsistent detailing of the sprinkler branch line at Mechanical Room 3B between Fire Alarm Riser Diagram FA-501 and Fire Protection Sheet FP-101.'
- **referenced_documents**: annotator_b=['FA-501', 'FP-101']; gold=['Fire Alarm Riser Diagram FA-501', 'Fire Protection Sheet FP-101']
- **deadline_text**: annotator_b='none stated'; gold="There's no hard deadline on this, but a timely response is appreciated."
- **assigned_reviewer**: annotator_b='Plumbing/Fire Protection Engineer'; gold="Owner's Representative"
- **routing_rationale**: annotator_b='The primary discrepancy involves fire suppression system detailing, requiring the Fire Protection engineer to reconcile the conflicting documents and issue corrected guidance before sprinkler shop drawings proceed.'; gold="Simultaneous cost and schedule impact on this project routes first to the Owner's Representative per the delegation-of-authority letter, with the Plumbing/Fire Protection Engineer looped in as technical reviewer."

## dev-0004
- **question_summary**: annotator_b='Existing electrical feeder condition at Electrical Closet 2A not shown on design documents (E-601, E-501) conflicts with planned work; clarification needed on whether reroute requires revised panel schedule.'; gold='Unforeseen existing feeder condition at Electrical Closet 2A conflicting with Electrical Panel Schedule E-601 and Sheet E-501.'
- **referenced_documents**: annotator_b=['Electrical Panel Schedule E-601', 'Sheet E-501', 'Project Engineer sketch (attachment)']; gold=['Electrical Panel Schedule E-601', 'Sheet E-501']
- **proposed_solution**: annotator_b='Reroute per attached sketch to clear the existing feeder condition at Electrical Closet 2A'; gold='—'
- **deadline_text**: annotator_b='none stated'; gold='There is no specific deadline for this, but a timely response is appreciated.'
- **routing_rationale**: annotator_b='Electrical Engineer must confirm whether the proposed reroute requires formal documentation updates to the panel schedule and validate the reroute solution against electrical code and design standards.'; gold='Default discipline routing: the Electrical Engineer resolves field condition conflicts affecting electrical work.'

## dev-0005
- **secondary_disciplines**: annotator_b=['Mechanical']; gold=[]
- **question_summary**: annotator_b='Conduit routing conflict discovered in Mechanical Room 3B during field investigation, not depicted on Electrical Detail 2/E502 or One-Line Diagram E-002; requires clarification on proper routing to coordinate with mechanical systems.'; gold='Unforeseen conduit routing conflict at Mechanical Room 3B conflicting with Electrical Detail 2/E502 and One-Line Diagram E-002.'
- **deadline_text**: annotator_b='end of week'; gold='by end of week'
- **routing_rationale**: annotator_b='Electrical Engineer must resolve the conduit routing conflict shown in their design documents and coordinate with Mechanical to establish conflict-free routing in the field.'; gold='Default discipline routing: the Electrical Engineer resolves field condition conflicts affecting electrical work.'
- **deadline_iso**: annotator_b=None; gold='2025-03-14'
- **deadline_resolution_status**: annotator_b='unresolved'; gold='resolved_to_date'

## dev-0006
- **question_summary**: annotator_b='Conduit routing conflict discovered in field at Electrical Closet 2A not shown on Sheet E-501 or Panel Schedule E-601; clarification needed on whether reroute requires revised drawings.'; gold='Unforeseen conduit routing conflict at Electrical Closet 2A conflicting with Sheet E-501 and Electrical Panel Schedule E-601.'
- **proposed_solution**: annotator_b='Reroute per attached sketch to clear the conduit routing conflict at Electrical Closet 2A'; gold='—'
- **deadline_text**: annotator_b='Please respond by end of week so we can plan the rough-in sequence.'; gold='by end of week'
- **routing_rationale**: annotator_b="Field-discovered electrical conduit routing conflict at a mechanical closet requires expedited review and resolution by the project's Electrical Engineer to prevent downstream rough-in sequencing delays."; gold='Default discipline routing: the Electrical Engineer resolves field condition conflicts affecting electrical work.'
- **deadline_iso**: annotator_b=None; gold='2025-05-16'
- **deadline_resolution_status**: annotator_b='unresolved'; gold='resolved_to_date'

## dev-0007
- **primary_discipline**: annotator_b='Architectural'; gold='General'
- **csi_division**: annotator_b='08 00 00 (Openings)'; gold='—'
- **question_summary**: annotator_b='Clarification needed on whether the egress width requirement in Project Specifications Division 00 complies with the adopted code edition referenced in Section 23 05 00.'; gold='Whether the egress width requirement shown in Project Specifications Division 00 complies with the adopted code.'
- **referenced_documents**: annotator_b=['Project Specifications Division 00', 'Section 23 05 00']; gold=['Project Specifications Division 00']
- **deadline_text**: annotator_b='none stated'; gold='There is no specific deadline for this, but a timely response is appreciated.'
- **assigned_reviewer**: annotator_b='Architect of Record'; gold='Code Official/AHJ Liaison'
- **routing_rationale**: annotator_b='Egress width compliance is a core architectural and life safety design decision requiring the Architect of Record to confirm design intent against the adopted code edition and coordinate with any code official review.'; gold='All code compliance questions route to the Code Official/AHJ Liaison regardless of discipline.'
- **escalation**: annotator_b=True; gold=False

## dev-0008
- **rfi_type**: annotator_b='document_discrepancy'; gold='design_clarification'
- **secondary_disciplines**: annotator_b=['Plumbing']; gold=[]
- **question_summary**: annotator_b='Fire department connection (FDC) piping detailing in Fire Protection Detail 3/FP502 is inconsistent with Fire Sprinkler Riser Diagram FP-501; clarification needed before sprinkler shop drawings release and hydrostatic testing.'; gold='Inconsistent detailing of the fire department connection (FDC) piping at Mechanical Room 3B between Fire Protection Detail 3/FP502 and Fire Sprinkler Riser Diagram FP-501.'
- **deadline_text**: annotator_b='timely response appreciated; before underground flush and hydrostatic test'; gold="There's no hard deadline on this, but a timely response is appreciated."
- **routing_rationale**: annotator_b='Fire protection system design and FDC detailing require specialized expertise from the fire protection engineer or plumbing/fire protection engineer who designed these systems and can reconcile the two referenced drawings.'; gold='Default discipline routing: the Plumbing/Fire Protection Engineer resolves design clarifications affecting fire protection work.'
- **escalation**: annotator_b=True; gold=False

## dev-0009
- **secondary_disciplines**: annotator_b=['Structural']; gold=[]
- **csi_division**: annotator_b='31 00 00 (Earthwork)'; gold='33 00 00 (Utilities)'
- **question_summary**: annotator_b='Discrepancy between Grading Plan C-201 and Utility Plan C-301 regarding site retaining wall elevation at Pour 3; field verification confirms C-301 is correct pending survey benchmark correction.'; gold='Discrepancy between Grading Plan C-201 and Utility Plan C-301 on the site retaining wall condition at Pour 3.'
- **referenced_documents**: annotator_b=['C-201', 'C-301']; gold=['Grading Plan C-201', 'Utility Plan C-301']
- **proposed_solution**: annotator_b='Use elevation shown in Utility Plan C-301; Grading Plan C-201 to be corrected by addendum.'; gold='—'
- **deadline_text**: annotator_b='at your earliest convenience'; gold='Please respond at your earliest convenience.'
- **routing_rationale**: annotator_b='Civil Engineer is the primary discipline for earthwork and grading plans (31 00 00), and must verify which plan is authoritative and issue corrective addendum.'; gold='Default discipline routing: the Civil Engineer resolves document discrepancies affecting site/civil work.'

## dev-0010
- **urgency**: annotator_b='priority'; gold='critical'
- **question_summary**: annotator_b='Shear wall hold-down location at Gridline B/2 conflicts between Sheet S-301 and Structural Detail 3/S502 due to mechanical ductwork and architectural ceiling framing interference; clarification needed on which drawing governs and whether detail revision is required.'; gold='Conflict between Sheet S-301 and Structural Detail 3/S502 over the shear wall hold-down location at Gridline B/2.'
- **proposed_solution**: annotator_b='Revise the shear wall hold-down per Structural Detail 3/S502 and coordinate final dimensions with affected trades before fabrication.'; gold='—'
- **routing_rationale**: annotator_b='Structural Engineer must resolve the drawing conflict and issue a definitive detail revision while coordinating with mechanical and architectural trades to ensure constructability.'; gold='Default discipline routing: the Structural Engineer resolves this coordination conflict; critical urgency escalates the review but does not change who owns it.'

## dev-0011
- **rfi_type**: annotator_b='document_discrepancy'; gold='design_clarification'
- **question_summary**: annotator_b='Curtain wall panel head/jamb detail at Gridline E/9 differs between Sheet A-201 and Sheet A-301; clarification needed for shop drawing sequencing.'; gold='Conflicting head/jamb detail for the curtain wall panel at Gridline E/9 between Sheet A-201 and Sheet A-301.'
- **proposed_solution**: annotator_b='Use the condition shown in Sheet A-301; Sheet A-201 will be superseded by upcoming addendum.'; gold='—'
- **deadline_text**: annotator_b='Respond whenever convenient'; gold='Respond whenever convenient.'
- **routing_rationale**: annotator_b='Architectural drawing discrepancy requiring clarification of curtain wall detail hierarchy and confirmation of which sheet governs; Architect of Record must resolve conflicting documentation and confirm addendum status.'; gold='Default discipline routing: the Architect of Record interprets and clarifies the Contract Documents for this Architectural-discipline design clarification.'

## dev-0012
- **question_summary**: annotator_b='Request to substitute a linear slot alternate diffuser for the specified perforated face diffuser in HVAC system per Section 23 82 00.'; gold='Request to substitute a linear slot alternate diffuser in lieu of the specified perforated face diffuser, specified in Section 23 82 00.'
- **referenced_documents**: annotator_b=['Section 23 82 00']; gold=[]
- **proposed_solution**: annotator_b='Linear slot alternate diffuser in lieu of specified perforated face diffuser'; gold='—'
- **deadline_text**: annotator_b='Please respond Thursday so we can place the order.'; gold='Thursday'
- **routing_rationale**: annotator_b='Mechanical Engineer must review the technical equivalency of the alternate diffuser to ensure performance, airflow characteristics, and noise criteria match the original specification.'; gold='Default discipline routing: the Mechanical Engineer evaluates substitution requests with no cost impact.'
- **deadline_iso**: annotator_b=None; gold='2025-09-25'
- **deadline_resolution_status**: annotator_b='unresolved'; gold='resolved_to_date'

## dev-0013
- **rfi_type**: annotator_b='document_discrepancy'; gold='design_clarification'
- **question_summary**: annotator_b='Standpipe connection detailing at Lobby 101 is inconsistent between Fire Sprinkler Riser Diagram FP-501 and Fire Protection Detail 3/FP-502; clarification needed before sprinkler shop drawings release.'; gold='Inconsistent detailing of the standpipe connection at Lobby 101 between Fire Sprinkler Riser Diagram FP-501 and Fire Protection Detail 3/FP502.'
- **referenced_documents**: annotator_b=['Fire Sprinkler Riser Diagram FP-501', 'Fire Protection Detail 3/FP-502']; gold=['Fire Sprinkler Riser Diagram FP-501', 'Fire Protection Detail 3/FP502']
- **proposed_solution**: annotator_b='Layout the standpipe connection per Fire Protection Detail 3/FP-502; disregard the conflicting note in Fire Sprinkler Riser Diagram FP-501.'; gold='—'
- **deadline_text**: annotator_b='none stated'; gold="There's no hard deadline on this, but a timely response is appreciated."
- **routing_rationale**: annotator_b='Fire protection system design authority must resolve the discrepancy between the two contract documents and authorize the correct detailing before shop drawings proceed.'; gold='Default discipline routing: the Plumbing/Fire Protection Engineer resolves design clarifications affecting fire protection work.'

## dev-0014
- **question_summary**: annotator_b='Request to substitute a linear slot alternate diffuser for the specified perforated face diffuser in HVAC system due to lead time concerns.'; gold='Request to substitute a linear slot alternate diffuser in lieu of the specified perforated face diffuser, specified in Section 23 30 00.'
- **referenced_documents**: annotator_b=['Section 23 30 00']; gold=[]
- **proposed_solution**: annotator_b='Substitute linear slot alternate diffuser; no schedule impact anticipated; potential cost credit or addition pending manufacturer pricing.'; gold='—'
- **assigned_reviewer**: annotator_b='Mechanical Engineer'; gold='Cost/Change-Order Manager'
- **routing_rationale**: annotator_b='Mechanical Engineer must evaluate product equivalency, performance characteristics, and compatibility with the specified HVAC system design before approval.'; gold='Substitution requests with a cost impact route to the Cost/Change-Order Manager to evaluate the credit or added cost before technical approval.'

## dev-0015
- **secondary_disciplines**: annotator_b=['Structural']; gold=[]
- **csi_division**: annotator_b='31 00 00 (Earthwork)'; gold='33 00 00 (Utilities)'
- **urgency**: annotator_b='priority'; gold='urgent'
- **question_summary**: annotator_b='Elevation discrepancy between Grading Plan C-201 and Sheet C-101 for site retaining wall at Placement Zone 2; resolved by field verification and survey benchmark correction.'; gold='Discrepancy between Grading Plan C-201 and Sheet C-101 on the site retaining wall condition at Placement Zone 2.'
- **proposed_solution**: annotator_b='Use elevation shown in Sheet C-101; Grading Plan C-201 to be corrected by addendum; site retaining wall matches C-101 once survey benchmark is corrected.'; gold='—'
- **deadline_text**: annotator_b='at your earliest convenience'; gold='Please respond at your earliest convenience.'
- **routing_rationale**: annotator_b='Civil Engineer leads earthwork and site grading design; Structural Engineer involvement secondary for retaining wall structural verification.'; gold='Default discipline routing: the Civil Engineer resolves this document discrepancy; urgent status escalates the review but does not change who owns it.'
- **escalation**: annotator_b=False; gold=True

## dev-0016
- **rfi_type**: annotator_b='document_discrepancy'; gold='design_clarification'
- **question_summary**: annotator_b='Hollow metal door assembly detail at Gridline D/7 shows conflicting head/jamb conditions between Detail 5/A501 and Sheet A-201 requirements.'; gold='Conflicting head/jamb detail for the hollow metal door assembly at Gridline D/7 between Detail 5/A501 and Sheet A-201.'
- **deadline_text**: annotator_b='none stated'; gold='No specific response date is requested.'
- **routing_rationale**: annotator_b='Discrepancy between architectural details requires the Architect of Record to reconcile conflicting head/jamb conditions and issue clarified door assembly specifications.'; gold='Default discipline routing: the Architect of Record interprets and clarifies the Contract Documents for this Architectural-discipline design clarification.'

## dev-0017
- **rfi_type**: annotator_b='document_discrepancy'; gold='design_clarification'
- **question_summary**: annotator_b='Curtain wall panel head/jamb detail at Gridline B/2 conflicts between Sheet A-301 and Detail 5/A501; clarification needed on correct condition.'; gold='Conflicting head/jamb detail for the curtain wall panel at Gridline B/2 between Sheet A-301 and Detail 5/A501.'
- **deadline_text**: annotator_b='Please respond within five days so we can stay on schedule.'; gold='within five days'
- **routing_rationale**: annotator_b='Curtain wall details are fundamental architectural design elements; the AoR must reconcile the conflicting sheets and issue a clarification.'; gold='Default discipline routing: the Architect of Record interprets and clarifies the Contract Documents for this Architectural-discipline design clarification.'
- **deadline_iso**: annotator_b=None; gold='2025-09-11'
- **deadline_resolution_status**: annotator_b='unresolved'; gold='resolved_to_date'

## dev-0018
- **rfi_type**: annotator_b='document_discrepancy'; gold='design_clarification'
- **question_summary**: annotator_b='Domestic water riser detailing at Room 214 conflicts between Plumbing Fixture Schedule P-602 and Plumbing Detail 6/P503; clarification needed on which document governs the rough-in installation.'; gold='Inconsistent detailing of the domestic water riser at Room 214 between Plumbing Fixture Schedule P-602 and Plumbing Detail 6/P503.'
- **proposed_solution**: annotator_b='Locate the domestic water riser per Plumbing Detail 6/P503; disregard the conflicting note in Plumbing Fixture Schedule P-602.'; gold='—'
- **deadline_text**: annotator_b='none stated'; gold='There is no specific deadline for this, but a timely response is appreciated.'
- **routing_rationale**: annotator_b='The Plumbing/Fire Protection Engineer is responsible for resolving conflicts between plumbing construction documents and approving which detail governs installation before field rough-in proceeds.'; gold='Default discipline routing: the Plumbing/Fire Protection Engineer resolves design clarifications affecting plumbing work.'

## dev-0019
- **question_summary**: annotator_b='Request to substitute a linear slot alternate diffuser for the specified perforated face diffuser in Section 23 82 00 due to lead time concerns.'; gold='Request to substitute a linear slot alternate diffuser in lieu of the specified perforated face diffuser, specified in Section 23 82 00.'
- **referenced_documents**: annotator_b=['Section 23 82 00']; gold=[]
- **proposed_solution**: annotator_b='Linear slot alternate diffuser as substitute for specified perforated face diffuser'; gold='—'
- **routing_rationale**: annotator_b='Mechanical Engineer must evaluate the performance equivalency, airflow characteristics, noise levels, and aesthetic compatibility of the proposed linear slot diffuser substitute against the specification.'; gold='Default discipline routing: the Mechanical Engineer evaluates substitution requests with no cost impact.'

## dev-0020
- **rfi_type**: annotator_b='document_discrepancy'; gold='design_clarification'
- **question_summary**: annotator_b='Curtain wall panel detail at Gridline C/4 shown in Sheet A-301 conflicts with head condition specified in Door and Window Schedule A-601; clarification needed to proceed with shop drawings.'; gold='Conflicting head/jamb detail for the curtain wall panel at Gridline C/4 between Sheet A-301 and Door and Window Schedule A-601.'
- **schedule_impact**: annotator_b=True; gold=False
- **routing_rationale**: annotator_b='The Architect of Record must reconcile the conflicting details between their own drawings and schedules to provide an authoritative clarification on the correct curtain wall head condition.'; gold='Default discipline routing: the Architect of Record interprets and clarifies the Contract Documents for this Architectural-discipline design clarification.'

## dev-0021
- **secondary_disciplines**: annotator_b=['General']; gold=[]
- **urgency**: annotator_b='priority'; gold='urgent'
- **question_summary**: annotator_b='Conduit routing conflict discovered in field at Lobby 101 not shown on Electrical Detail 2/E502 or One-Line Diagram E-002; clarification needed on whether reroute requires revised electrical detail.'; gold='Unforeseen conduit routing conflict at Lobby 101 conflicting with Electrical Detail 2/E502 and One-Line Diagram E-002.'
- **proposed_solution**: annotator_b='Reroute per attached sketch to clear the conduit routing conflict at Lobby 101'; gold='—'
- **deadline_text**: annotator_b='end of day'; gold='by end of day'
- **assigned_reviewer**: annotator_b='Electrical Engineer'; gold='GC Superintendent'
- **routing_rationale**: annotator_b='Electrical Engineer must review the proposed conduit reroute sketch, confirm routing feasibility, and determine if design documentation updates are required for compliance and future reference.'; gold="Field condition conflicts that are both urgent and schedule-impacting route directly to the GC Superintendent for immediate field resolution, unless the delegation-of-authority letter already pulled this to the Owner's Representative for a simultaneous cost impact."
- **escalation**: annotator_b=False; gold=True
- **deadline_iso**: annotator_b=None; gold='2025-03-06'
- **deadline_resolution_status**: annotator_b='unresolved'; gold='resolved_to_date'

## dev-0022
- **question_summary**: annotator_b='Does the concrete cover requirement at the footing in Structural Detail 3/S502 comply with the code edition referenced in Section 03 20 00?'; gold='Whether the concrete cover requirement at the footing shown in Structural Detail 3/S502 meets the applicable code.'
- **referenced_documents**: annotator_b=['Structural Detail 3/S502', 'Section 03 20 00']; gold=['Structural Detail 3/S502']
- **answer_in_documents**: annotator_b=False; gold=True
- **assigned_reviewer**: annotator_b='Structural Engineer'; gold='Code Official/AHJ Liaison'
- **routing_rationale**: annotator_b='The Structural Engineer must verify whether the specified concrete cover in Detail 3/S502 complies with the applicable code edition referenced in the specifications.'; gold='All code compliance questions route to the Code Official/AHJ Liaison regardless of discipline; the answer is already present in the referenced documents.'

## dev-0023
- **rfi_type**: annotator_b='document_discrepancy'; gold='coordination_conflict'
- **csi_division**: annotator_b='03 00 00 (Concrete)'; gold='05 00 00 (Metals)'
- **question_summary**: annotator_b='Embed plate location at Gridline C/4 conflicts between Structural General Notes S-001 and Foundation Plan S-101, creating coordination issues with mechanical ductwork and architectural ceiling framing; fabrication cutoff imminent.'; gold='Conflict between Structural General Notes S-001 and Foundation Plan S-101 over the embed plate location at Gridline C/4.'
- **referenced_documents**: annotator_b=['S-001', 'S-101']; gold=['Structural General Notes S-001', 'Foundation Plan S-101']
- **proposed_solution**: annotator_b='Revise embed plate per Foundation Plan S-101 and coordinate final dimensions with affected trades before fabrication.'; gold='—'
- **deadline_text**: annotator_b="before the fabricator's shop drawing cutoff"; gold="Please respond before the fabricator's shop drawing cutoff."
- **assigned_reviewer**: annotator_b='Structural Engineer'; gold="Owner's Representative"
- **routing_rationale**: annotator_b='Structural Engineer must resolve the internal document conflict between S-001 and S-101 and coordinate with Mechanical and Architectural disciplines on the embed plate location to avoid fabrication delays.'; gold="Simultaneous cost and schedule impact on this project routes first to the Owner's Representative per the delegation-of-authority letter, with the Structural Engineer looped in as technical reviewer."

## dev-0024
- **csi_division**: annotator_b='32 00 00 (Exterior Improvements)'; gold='33 00 00 (Utilities)'
- **question_summary**: annotator_b='Storm drainage inlet elevation discrepancy between Sheet C-101 and Utility Plan C-301 near Pour 3 requires clarification on which elevation is correct.'; gold='Discrepancy between Sheet C-101 and Utility Plan C-301 on the storm drainage inlet condition at Pour 3.'
- **proposed_solution**: annotator_b='Use the elevation shown in Utility Plan C-301; Sheet C-101 to be corrected by addendum.'; gold='—'
- **deadline_text**: annotator_b='none stated'; gold='Let us know when you expect to have an answer on this.'
- **routing_rationale**: annotator_b='Civil Engineer must resolve the utility/drainage elevation discrepancy between the two construction documents and issue corrective addendum.'; gold='Default discipline routing: the Civil Engineer resolves document discrepancies affecting site/civil work.'

## dev-0025
- **rfi_type**: annotator_b='document_discrepancy'; gold='design_clarification'
- **question_summary**: annotator_b='Backflow preventer assembly detailing in Sheet P-601 Room 214 is inconsistent with Plumbing Riser Diagram P-501; clarification needed before rough-in proceeds.'; gold='Inconsistent detailing of the backflow preventer assembly at Room 214 between Sheet P-601 and Plumbing Riser Diagram P-501.'
- **deadline_text**: annotator_b='none stated'; gold='No hard deadline, just would like to keep this moving.'
- **assigned_reviewer**: annotator_b='Plumbing/Fire Protection Engineer'; gold="Owner's Representative"
- **routing_rationale**: annotator_b='Backflow preventer assembly conflicts require the plumbing design engineer to reconcile sheet details with riser diagrams and confirm constructability before rough-in proceeds.'; gold="Simultaneous cost and schedule impact on this project routes first to the Owner's Representative per the delegation-of-authority letter, with the Plumbing/Fire Protection Engineer looped in as technical reviewer."
- **escalation**: annotator_b=False; gold=True

## dev-0026
- **urgency**: annotator_b='priority'; gold='urgent'
- **question_summary**: annotator_b='Electrical closet panel clearance obstruction discovered in field conflicts with One-Line Diagram E-002 and Electrical Panel Schedule E-601; clarification requested on whether reroute requires revised documentation.'; gold='Unforeseen panel clearance obstruction at Electrical Closet 2A conflicting with One-Line Diagram E-002 and Electrical Panel Schedule E-601.'
- **proposed_solution**: annotator_b='Reroute per attached sketch to clear the panel clearance obstruction.'; gold='—'
- **deadline_text**: annotator_b='end of week'; gold='by end of week'
- **routing_rationale**: annotator_b='The Electrical Engineer must verify whether the field reroute solution requires formal drawing revisions and confirm compliance with electrical code clearance requirements and coordination with the one-line diagram.'; gold='Default discipline routing: the Electrical Engineer resolves this field condition; urgent status escalates the review but does not change who owns it since there is no schedule impact.'
- **escalation**: annotator_b=False; gold=True
- **deadline_iso**: annotator_b=None; gold='2025-05-02'
- **deadline_resolution_status**: annotator_b='unresolved'; gold='resolved_to_date'

## dev-0027
- **rfi_type**: annotator_b='design_clarification'; gold='code_compliance_question'
- **question_summary**: annotator_b='Verification that the minimum reinforcing lap length for the shear wall assembly in Foundation Plan S-101 complies with the code edition referenced in Section 03 20 00.'; gold='Whether the minimum reinforcing lap length for the shear wall assembly shown in Foundation Plan S-101 meets the applicable code.'
- **referenced_documents**: annotator_b=['Foundation Plan S-101', 'Detail 7/S301', 'Section 03 20 00']; gold=['Foundation Plan S-101']
- **assigned_reviewer**: annotator_b='Structural Engineer'; gold='Code Official/AHJ Liaison'
- **routing_rationale**: annotator_b='The Structural Engineer must verify code compliance for reinforcing lap lengths in shear wall design and confirm that the existing detail meets applicable building code requirements.'; gold='All code compliance questions route to the Code Official/AHJ Liaison regardless of discipline; the answer is already present in the referenced documents.'

## dev-0028
- **urgency**: annotator_b='urgent'; gold='critical'
- **question_summary**: annotator_b='Cast-in anchor bolt group location shown in Foundation Plan S-101 at Gridline B/2 conflicts with mechanical ductwork and architectural ceiling framing routing shown in Structural Detail 3/S502; resolution needed before steel fabrication release.'; gold='Conflict between Foundation Plan S-101 and Structural Detail 3/S502 over the cast-in anchor bolt group location at Gridline B/2.'
- **proposed_solution**: annotator_b='Revise the cast-in anchor bolt group per Structural Detail 3/S502 and coordinate final dimensions with affected trades before fabrication.'; gold='—'
- **deadline_text**: annotator_b='before the next steel erection mobilization'; gold='We need this resolved before the next steel erection mobilization.'
- **routing_rationale**: annotator_b='Structural Engineer must resolve the foundation plan conflict and coordinate with mechanical and architectural trades to ensure cast-in anchor bolt placement does not interfere with ductwork and ceiling framing.'; gold='Default discipline routing: the Structural Engineer resolves this coordination conflict; critical urgency escalates the review but does not change who owns it.'

## dev-0029
- **question_summary**: annotator_b='Request to substitute a fan-coil-based alternate air handling unit for the specified packaged rooftop AHU due to lead time constraints.'; gold='Request to substitute a fan-coil-based alternate air handling unit in lieu of the specified packaged rooftop AHU, specified in Section 23 05 00.'
- **referenced_documents**: annotator_b=['Section 23 05 00']; gold=[]
- **proposed_solution**: annotator_b='Substitute fan-coil-based alternate AHU with performance data confirming equivalence to Section 23 05 00 specifications.'; gold='—'
- **routing_rationale**: annotator_b='Mechanical Engineer must evaluate performance equivalence of the proposed fan-coil alternate against the specified rooftop AHU requirements in Section 23 05 00.'; gold='Default discipline routing: the Mechanical Engineer evaluates substitution requests with no cost impact.'

## dev-0030
- **primary_discipline**: annotator_b='Architectural'; gold='General'
- **csi_division**: annotator_b='07 00 00 (Thermal and Moisture Protection)'; gold='—'
- **urgency**: annotator_b='priority'; gold='urgent'
- **question_summary**: annotator_b='Confirmation needed on whether the occupancy separation rating in Project Specifications Division 00 complies with the adopted code edition referenced in Section 07 92 00.'; gold='Whether the occupancy separation rating shown in Project Specifications Division 00 complies with the adopted code.'
- **referenced_documents**: annotator_b=['Project Specifications Division 00', 'Section 07 92 00']; gold=['Project Specifications Division 00']
- **deadline_text**: annotator_b='none stated'; gold='No hard deadline, just would like to keep this moving.'
- **assigned_reviewer**: annotator_b='Architect of Record'; gold='Code Official/AHJ Liaison'
- **routing_rationale**: annotator_b='The Architect of Record must verify occupancy separation ratings against building code compliance and confirm design intent, as this is a fundamental code-driven architectural specification.'; gold='All code compliance questions route to the Code Official/AHJ Liaison regardless of discipline; urgent status escalates the review.'
- **escalation**: annotator_b=False; gold=True

## dev-0031
- **rfi_type**: annotator_b='document_discrepancy'; gold='design_clarification'
- **secondary_disciplines**: annotator_b=['Electrical']; gold=[]
- **question_summary**: annotator_b='Fire pump discharge piping detailing is inconsistent between Fire Protection Detail 3/FP502 and Fire Alarm Riser Diagram FA-501 at Electrical Closet 2A; clarification needed before sprinkler shop drawing release.'; gold='Inconsistent detailing of the fire pump discharge piping at Electrical Closet 2A between Fire Protection Detail 3/FP502 and Fire Alarm Riser Diagram FA-501.'
- **deadline_text**: annotator_b='none stated'; gold="There's no hard deadline on this, but a timely response is appreciated."
- **routing_rationale**: annotator_b='The primary issue concerns fire protection system detailing and coordination; the Plumbing/Fire Protection Engineer must reconcile the conflicting fire pump discharge piping details and coordinate with the electrical riser diagram.'; gold='Default discipline routing: the Plumbing/Fire Protection Engineer resolves design clarifications affecting fire protection work.'

## dev-0032
- **primary_discipline**: annotator_b='Fire_Protection'; gold='General'
- **secondary_disciplines**: annotator_b=['Structural']; gold=[]
- **csi_division**: annotator_b='21 00 00 (Fire Suppression)'; gold='—'
- **question_summary**: annotator_b='Confirmation that the fire-rated assembly listing in Project Specifications Division 00 complies with the adopted code edition referenced in Section 03 30 00.'; gold='Whether the fire-rated assembly listing shown in Project Specifications Division 00 complies with the adopted code.'
- **referenced_documents**: annotator_b=['Project Specifications Division 00', 'Section 03 30 00']; gold=['Project Specifications Division 00']
- **deadline_text**: annotator_b='none stated'; gold='No hard deadline, just would like to keep this moving.'
- **routing_rationale**: annotator_b='Fire-rated assembly code compliance requires verification against adopted code edition and AHJ interpretation; response from Project Engineer confirms compliance but formal code review by AHJ liaison ensures regulatory standing.'; gold='All code compliance questions route to the Code Official/AHJ Liaison regardless of discipline.'

## dev-0033
- **rfi_type**: annotator_b='document_discrepancy'; gold='design_clarification'
- **question_summary**: annotator_b='Backflow preventer assembly detailing is inconsistent between Plumbing Fixture Schedule P-602 and Plumbing Detail 6/P503 at Electrical Closet 2A; clarification needed before rough-in inspection.'; gold='Inconsistent detailing of the backflow preventer assembly at Electrical Closet 2A between Plumbing Fixture Schedule P-602 and Plumbing Detail 6/P503.'
- **deadline_text**: annotator_b='ahead of the next rough-in inspection'; gold='There is no specific deadline for this, but a timely response is appreciated.'
- **routing_rationale**: annotator_b='The Plumbing Engineer is best positioned to reconcile the conflicting detailing between the fixture schedule and construction detail for the backflow preventer assembly.'; gold='Default discipline routing: the Plumbing/Fire Protection Engineer resolves design clarifications affecting plumbing work.'

## dev-0034
- **secondary_disciplines**: annotator_b=['Structural']; gold=[]
- **csi_division**: annotator_b='31 00 00 (Earthwork)'; gold='33 00 00 (Utilities)'
- **question_summary**: annotator_b='Discrepancy between Utility Plan C-301 and Sheet C-101 regarding site retaining wall elevation at Pour 3; field verification confirms C-101 is correct pending survey benchmark correction.'; gold='Discrepancy between Utility Plan C-301 and Sheet C-101 on the site retaining wall condition at Pour 3.'
- **proposed_solution**: annotator_b='Use the elevation shown in Sheet C-101; Utility Plan C-301 will be corrected by addendum.'; gold='—'
- **deadline_text**: annotator_b='A response next week would be appreciated.'; gold='next week'
- **routing_rationale**: annotator_b='The discrepancy involves site retaining wall elevations and survey benchmarks, which are primary civil engineering concerns; Structural coordination is secondary.'; gold='Default discipline routing: the Civil Engineer resolves document discrepancies affecting site/civil work.'
- **deadline_iso**: annotator_b=None; gold='2025-03-14'
- **deadline_resolution_status**: annotator_b='unresolved'; gold='resolved_to_date'

## dev-0035
- **secondary_disciplines**: annotator_b=['Structural']; gold=[]
- **csi_division**: annotator_b='31 00 00 (Earthwork)'; gold='33 00 00 (Utilities)'
- **question_summary**: annotator_b='Elevation discrepancy between Grading Plan C-201 and Utility Plan C-301 for site retaining wall at Pour Sequence 4; field verification confirms C-301 is correct pending survey benchmark correction.'; gold='Discrepancy between Grading Plan C-201 and Utility Plan C-301 on the site retaining wall condition at Pour Sequence 4.'
- **proposed_solution**: annotator_b='Use elevation shown in Utility Plan C-301 for site retaining wall at Pour Sequence 4; Grading Plan C-201 to be corrected by addendum.'; gold='—'
- **deadline_text**: annotator_b='at your earliest convenience'; gold='Please respond at your earliest convenience.'
- **routing_rationale**: annotator_b='Civil Engineer is responsible for site grading, utility coordination, and retaining wall elevations as primary discipline; Structural Engineer secondarily involved due to retaining wall structural implications.'; gold='Default discipline routing: the Civil Engineer resolves document discrepancies affecting site/civil work.'

## dev-0036
- **question_summary**: annotator_b='Does the minimum reinforcing lap length for the shear wall assembly in Structural General Notes S-001 comply with the code edition referenced in Section 03 15 00?'; gold='Whether the minimum reinforcing lap length for the shear wall assembly shown in Structural General Notes S-001 meets the applicable code.'
- **referenced_documents**: annotator_b=['Structural General Notes S-001', 'Section 03 15 00']; gold=['Structural General Notes S-001']
- **answer_in_documents**: annotator_b=False; gold=True
- **deadline_text**: annotator_b='A response within three days would be appreciated.'; gold='within three days'
- **assigned_reviewer**: annotator_b='Structural Engineer'; gold='Code Official/AHJ Liaison'
- **routing_rationale**: annotator_b='This is a structural code compliance question regarding reinforcing lap length in concrete shear walls that requires the Structural Engineer to verify compliance with applicable building code requirements.'; gold='All code compliance questions route to the Code Official/AHJ Liaison regardless of discipline; the answer is already present in the referenced documents.'
- **deadline_iso**: annotator_b=None; gold='2025-10-30'
- **deadline_resolution_status**: annotator_b='unresolved'; gold='resolved_to_date'

## dev-0037
- **rfi_type**: annotator_b='document_discrepancy'; gold='design_clarification'
- **question_summary**: annotator_b='Sanitary vent stack detailing for Room 214 is inconsistent between Plumbing Fixture Schedule P-602 and Plumbing Riser Diagram P-501; clarification needed before rough-in inspection.'; gold='Inconsistent detailing of the sanitary vent stack at Room 214 between Plumbing Fixture Schedule P-602 and Plumbing Riser Diagram P-501.'
- **deadline_text**: annotator_b='none stated'; gold='No hard deadline, just would like to keep this moving.'
- **routing_rationale**: annotator_b='The Plumbing/Fire Protection Engineer must reconcile the conflicting vent stack details between the two plumbing documents and issue clarification before rough-in inspection proceeds.'; gold='Default discipline routing: the Plumbing/Fire Protection Engineer resolves design clarifications affecting plumbing work.'

## dev-0038
- **csi_division**: annotator_b='03 00 00 (Concrete)'; gold='05 00 00 (Metals)'
- **question_summary**: annotator_b='Cast-in anchor bolt group location at Gridline E/9 on Foundation Plan S-101 conflicts with mechanical ductwork and architectural ceiling framing routing on Sheet S-301; clarification needed on which plan controls before steel fabrication release.'; gold='Conflict between Foundation Plan S-101 and Sheet S-301 over the cast-in anchor bolt group location at Gridline E/9.'
- **proposed_solution**: annotator_b='Revise the cast-in anchor bolt group per Sheet S-301 and coordinate final dimensions with affected trades before fabrication.'; gold='—'
- **deadline_text**: annotator_b='before the next steel erection mobilization'; gold='We need this resolved before the next steel erection mobilization.'
- **routing_rationale**: annotator_b='The Structural Engineer must resolve the foundation anchor bolt conflict and coordinate with Mechanical and Architectural disciplines to establish a unified design before steel fabrication proceeds.'; gold='Default discipline routing: the Structural Engineer resolves this coordination conflict; critical urgency escalates the review but does not change who owns it.'

## dev-0039
- **rfi_type**: annotator_b='document_discrepancy'; gold='design_clarification'
- **urgency**: annotator_b='routine'; gold='priority'
- **question_summary**: annotator_b='Curtain wall panel head/jamb detail at Gridline C/4 differs between Detail 5/A501 and Sheet A-201; clarification needed on correct condition.'; gold='Conflicting head/jamb detail for the curtain wall panel at Gridline C/4 between Detail 5/A501 and Sheet A-201.'
- **deadline_text**: annotator_b='Respond whenever convenient'; gold='Respond whenever convenient.'
- **routing_rationale**: annotator_b='Document discrepancy in curtain wall details requires the Architect of Record to reconcile conflicting conditions between construction documents.'; gold='Default discipline routing: the Architect of Record interprets and clarifies the Contract Documents for this Architectural-discipline design clarification.'

## dev-0040
- **secondary_disciplines**: annotator_b=['Architectural']; gold=[]
- **question_summary**: annotator_b='Confirmation needed on whether the occupancy separation rating specified in Project Manual Section 01 00 00 complies with the code edition referenced in Section 26 05 00.'; gold='Whether the occupancy separation rating shown in Project Manual Section 01 00 00 complies with the adopted code.'
- **referenced_documents**: annotator_b=['Project Manual Section 01 00 00', 'Project Manual Section 26 05 00']; gold=['Project Manual Section 01 00 00']
- **answer_in_documents**: annotator_b=True; gold=False
- **deadline_text**: annotator_b='none stated'; gold='No hard deadline, just would like to keep this moving.'
- **routing_rationale**: annotator_b='Occupancy separation ratings are code-compliance matters requiring independent verification against the adopted code edition; the AHJ Liaison can definitively confirm compliance and resolve any ambiguity between design intent and code reading.'; gold='All code compliance questions route to the Code Official/AHJ Liaison regardless of discipline.'
