DECLARE

	@VarDefaultInt		Int			= (SELECT Value FROM dbo.GlobalVariables (NOLOCK) WHERE  GroupID = 2)
	,@VarDefaultDate	DateTime	= (SELECT Value FROM dbo.GlobalVariables (NOLOCK) WHERE  GroupID = 3)
	,@VarDefaultString	VarChar(20)	= (SELECT Value FROM dbo.GlobalVariables (NOLOCK) WHERE  GroupID = 4)
	,@SourceSystemID	Int			= (SELECT SourceSystemID FROM dbo.DimSourceSystem WHERE SourceSystem = 'RiO')

SELECT DISTINCT

	PathwayStage								=	CASE
														WHEN WL.WaitingListRuleName LIKE '%Screening%' THEN 'Triage'
														WHEN WL.WaitingListRuleName LIKE '%Initial%' THEN 'Assessment'
														WHEN WL.WaitingListRuleName LIKE '%Further%' OR WL.WaitingListRuleName LIKE '%Gather%' THEN 'School'
														WHEN WL.WaitingListRuleName LIKE '%Post%' THEN 'Post Diagnostic Pathways'
														WHEN WL.WaitingListRuleName LIKE '%Titrat%' THEN 'Titration'
														WHEN WL.WaitingListRuleName LIKE '%Await%' THEN 'Diagnosis'
														WHEN WL.WaitingListRuleName LIKE '%AMR%' OR WL.WaitingListRuleName LIKE '%Review%' THEN 'Awaiting Review'
														ELSE 'Not Known'
													END
	,WaitingWeeks								=	AVG(CAST(DATEDIFF(D,WL.WaitingStartDate,WL.WaitingRemovalDate)/7.0 AS numeric(5,1)))
	,WaitingDays								=   AVG(DATEDIFF(D,WL.WaitingStartDate,WL.WaitingRemovalDate))
	
FROM [Informatics_SSAS_Live].[dbo].[FactWaitingList] WL

		INNER JOIN Informatics_SSAS_Live.dbo.FactReferrals RF
		ON WL.ReferralID = RF.ReferralID

		INNER JOIN Informatics_SSAS_Live.dbo.DimTeam TM
		ON WL.TeamID = TM.TeamID

		INNER JOIN Informatics_SSAS_Live.dbo.DimPriority PR
		ON RF.PriorityID = PR.PriorityID

WHERE	WL.WaitingRemovalDate <> '1901-01-01'
		AND WL.WaitingStartDate >= '2023-04-01'
		AND WL.IsWaitingListEntryCancelled IN  (0, -999)
		AND WL.SourceSystemID = @SourceSystemID
		AND Priority = 'Routine'
		AND TM.TeamID IN (SELECT TeamID FROM Informatics_SSAS_Live.dbo.pvw_7593_CYPMHPerformanceDashboard_Teams TS WHERE TS.Team LIKE '%ADHD%' AND TS.Team LIKE '%(S)%')

GROUP BY

	CASE
		WHEN WL.WaitingListRuleName LIKE '%Screening%' THEN 'Triage'
		WHEN WL.WaitingListRuleName LIKE '%Initial%' THEN 'Assessment'
		WHEN WL.WaitingListRuleName LIKE '%Further%' OR WL.WaitingListRuleName LIKE '%Gather%' THEN 'School'
		WHEN WL.WaitingListRuleName LIKE '%Post%' THEN 'Post Diagnostic Pathways'
		WHEN WL.WaitingListRuleName LIKE '%Titrat%' THEN 'Titration'
		WHEN WL.WaitingListRuleName LIKE '%Await%' THEN 'Diagnosis'
		WHEN WL.WaitingListRuleName LIKE '%AMR%' OR WL.WaitingListRuleName LIKE '%Review%' THEN 'Awaiting Review'
		ELSE 'Not Known'
	END