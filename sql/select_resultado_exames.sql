
select 
          p.datanascimento 
      ,lsri.[idResultado]
      ,lsri.[idlancamentoservico]
      ,lsri.[MargemNormalMinimaH]
      ,lsri.[MargemNormalMaximaH]
      ,lsri.[MargemAbsurdaMinima]
      ,lsri.[MargemAbsurdaMaxima]
      ,lsri.[Grupo]
      ,lsri.[Setor]
      ,lsri.[Material]
      ,lsri.[MargemNormalMinimaM]
      ,lsri.[MargemNormalMaximaM]
      ,lsri.[Item]
      ,lsri.[Servico]
      ,lsri.[Alfanumerico]
      ,lsri.[Resultado] as [tempopararesultado]
      ,lsri.[Paciente]
      ,lsri.[PostoCliente]
      ,lsri.[PostoColeta]
      ,lsri.[Idade]
      ,lsri.[DataColeta]
      ,lsri.[Medico] as [medicosolicitante]
      ,lsri.[ExameMetodo]
      ,lsri.[ExameResultado]
      ,lsri.[ItemGrupo]
      ,lsri.[Referencia]
      ,lsri.[ServicoExameMetodo]
      ,lsri.[ServicoExameMaterial]
      ,lsri.[ExameObservacao]
      ,lsri.[DataLiberado]
      ,lsri.[resultadoobservacao]
      ,lsri.[Talao]
      ,lsri.[Sexo]
      ,lsri.[LaboratorioTecnicoNome]
      ,lsri.[LaboratorioTecnicoIdentificacao]
      ,lsri.[LaboratorioTecnicoCabecalho]
      ,lsri.[PodeImprimir]
      ,lsri.[VisivelNoLaudoImpresso]
      ,lsri.[SaoMarcos] as [processadopelolab_saomarcos]
      ,lsri.[MensagemLaboratorio]
      ,lsri.[Alvaro] as [processadopelolab_alvaro]
      ,lsri.[Conclusao]
      ,lsri.[Observacao]
      ,lsri.[Laboratorio]
      from vw_Cad_LancamentoServicoResultadoItem lsri
        left join vw_cad_paciente p on p.matriculal = lsri.matricula and lsri.paciente = p.nome
    where (0=0)
        and (LEN(lsri.exameResultado) > 0)
        and lsri.paciente = ? and p.DataNascimento = ? and lsri.dataliberado >= ?
        and lsri.desativado = 0
        order by lsri.dataliberado desc
        -- order by lsri.grupo, lsri.servico, lsri.grupopagina, lsri.grupopaginaordem



