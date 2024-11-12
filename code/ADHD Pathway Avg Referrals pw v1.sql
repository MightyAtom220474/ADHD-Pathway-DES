DECLARE

	@SourceSystemID	Int			= (SELECT SourceSystemID FROM Informatics_SSAS_Live.dbo.DimSourceSystem WHERE SourceSystem = 'RiO')

SELECT

	AVG(AcceptedRefs) AS AvgRefsPW

FROM

(

SELECT DISTINCT

	DT.WeekCommencing
	,Count(FactWaitingListID) AS AcceptedRefs
	
FROM [Informatics_SSAS_Live].[dbo].[FactWaitingList] WL

		INNER JOIN Informatics_SSAS_Live.dbo.FactReferrals RF
		ON WL.ReferralID = RF.ReferralID

		INNER JOIN Informatics_SSAS_Live.dbo.DimTeam TM
		ON WL.TeamID = TM.TeamID

		INNER JOIN Informatics_SSAS_Live.dbo.DimPriority PR
		ON RF.PriorityID = PR.PriorityID

		LEFT JOIN Informatics_SSAS_Live.dbo.DimDate DT
		ON WL.WaitingStartDate = DT.Date

WHERE	WL.WaitingRemovalDate <> '1901-01-01'
		AND WL.WaitingStartDate >= '2023-04-01'
		AND WL.IsWaitingListEntryCancelled IN  (0, -999)
		AND WL.SourceSystemID = @SourceSystemID
		AND Priority = 'Routine'
		AND WL.WaitingListRuleName LIKE '%Initial%'
		AND TM.TeamID IN (SELECT TeamID FROM Informatics_SSAS_Live.dbo.pvw_7593_CYPMHPerformanceDashboard_Teams TS WHERE TS.Team LIKE '%ADHD%' AND TS.Team LIKE '%(S)%')

GROUP BY

	DT.WeekCommencing) Sub